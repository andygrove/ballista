// Copyright 2020 Andy Grove
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Distributed query execution
//!
//! This code is EXPERIMENTAL and still under development

use std::{collections::HashMap, future::Future};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use super::execution_plans::{self, QueryStageExec, ShuffleReaderExec, UnresolvedShuffleExec};
use crate::client::BallistaClient;
use crate::context::DFTableAdapter;
use crate::error::{BallistaError, Result};
use crate::executor::collect::CollectExec;
use crate::serde::scheduler::ExecutorMeta;
use crate::serde::scheduler::PartitionId;
use crate::utils;

use crate::utils::format_plan;
use arrow::record_batch::RecordBatch;
use datafusion::error::DataFusionError;
use datafusion::execution::context::ExecutionContext;
use datafusion::logical_plan::LogicalPlan;
use datafusion::physical_plan::csv::CsvExec;
use datafusion::physical_plan::expressions::Column;
use datafusion::physical_plan::hash_aggregate::{AggregateMode, HashAggregateExec};
use datafusion::physical_plan::hash_join::HashJoinExec;
use datafusion::physical_plan::merge::MergeExec;
use datafusion::physical_plan::{
    AggregateExpr, ExecutionPlan, PhysicalExpr, SendableRecordBatchStream,
};
use log::{debug, info};
use uuid::Uuid;
use datafusion::physical_plan::coalesce_batches::CoalesceBatchesExec;
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::parquet::ParquetExec;
use datafusion::physical_plan::sort::SortExec;
use datafusion::physical_plan::projection::ProjectionExec;
use std::sync::atomic::{AtomicUsize, Ordering};


type SendableExecutionPlan = Pin<Box<dyn Future<Output = Result<Arc<dyn ExecutionPlan>>> + Send>>;
type PartialQueryStageResult = (Arc<dyn ExecutionPlan>, Vec<Arc<QueryStageExec>>);

#[derive(Debug, Clone)]
pub struct PartitionLocation {
    pub(crate) partition_id: PartitionId,
    pub(crate) executor_meta: ExecutorMeta,
}

pub struct DistributedPlanner {
    executors: Vec<ExecutorMeta>,
    next_stage_id: usize,
}

impl DistributedPlanner {
    pub fn try_new(executors: Vec<ExecutorMeta>) -> Result<Self> {
        if executors.is_empty() {
            Err(BallistaError::General(
                "DistributedPlanner requires at least one executor".to_owned(),
            ))
        } else {
            Ok(Self {
                executors,
                next_stage_id: 0,
            })
        }
    }
}

impl DistributedPlanner {
    /// Execute a distributed query against a cluster, leaving the final results on the
    /// executors. The [ExecutionPlan] returned by this method is guaranteed to be a
    /// [ShuffleReaderExec] that can be used to fetch the final results from the executors
    /// in parallel.
    pub async fn execute_distributed_query(
        &mut self,
        execution_plan: Arc<dyn ExecutionPlan>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let job_uuid = Uuid::new_v4();

        let now = Instant::now();
        let execution_plans = self.plan_query_stages(&job_uuid, execution_plan)?;

        // until we have a web UI, this gives us a way to view the query plans
        let dot_file = format!("query-{}.dot", job_uuid);
        produce_diagram(&dot_file, &execution_plans)?;
        info!("Query plan diagram written to {}", dot_file);

        info!(
            "DistributedPlanner created {} execution plans in {} seconds:",
            execution_plans.len(),
            now.elapsed().as_secs()
        );

        for plan in &execution_plans {
            info!("{}", format_plan(plan.as_ref(), 0)?);
        }

        execute(execution_plans, self.executors.clone()).await
    }

    /// Returns a vector of ExecutionPlans, where the root node is a [QueryStageExec].
    /// Plans that depend on the input of other plans will have leaf nodes of type [UnresolvedShuffleExec].
    /// A [QueryStageExec] is created whenever the partitioning changes.
    ///
    /// Returns an empty vector if the execution_plan doesn't need to be sliced into several stages.
    pub fn plan_query_stages(
        &mut self,
        job_uuid: &Uuid,
        execution_plan: Arc<dyn ExecutionPlan>,
    ) -> Result<Vec<Arc<QueryStageExec>>> {
        info!("planning query stages");
        let (new_plan, mut stages) = self.plan_query_stages_internal(job_uuid, execution_plan)?;
        stages.push(create_query_stage(
            job_uuid,
            self.next_stage_id(),
            new_plan,
        )?);
        Ok(stages)
    }

    /// Returns a potentially modified version of the input execution_plan along with the resulting query stages.
    /// This function is needed because the input execution_plan might need to be modified, but it might not hold a
    /// compelte query stage (its parent might also belong to the same stage)
    fn plan_query_stages_internal(
        &mut self,
        job_uuid: &Uuid,
        execution_plan: Arc<dyn ExecutionPlan>,
    ) -> Result<PartialQueryStageResult> {
        // recurse down and replace children
        if execution_plan.children().is_empty() {
            return Ok((execution_plan, vec![]));
        }

        let mut stages = vec![];
        let mut children = vec![];
        for child in execution_plan.children() {
            let (new_child, mut child_stages) =
                self.plan_query_stages_internal(&job_uuid, child.clone())?;
            children.push(new_child);
            stages.append(&mut child_stages);
        }

        if let Some(adapter) = execution_plan.as_any().downcast_ref::<DFTableAdapter>() {
            let ctx = ExecutionContext::new();
            Ok((ctx.create_physical_plan(&adapter.logical_plan)?, stages))
        } else if let Some(merge) = execution_plan.as_any().downcast_ref::<MergeExec>() {
            let query_stage =
                create_query_stage(job_uuid, self.next_stage_id(), merge.children()[0].clone())?;
            let unresolved_shuffle = Arc::new(UnresolvedShuffleExec::new(
                vec![query_stage.stage_id],
                query_stage.schema(),
                query_stage.output_partitioning().partition_count(),
            ));
            stages.push(query_stage);
            Ok((merge.with_new_children(vec![unresolved_shuffle])?, stages))
        } else if let Some(agg) = execution_plan.as_any().downcast_ref::<HashAggregateExec>() {
            //TODO should insert query stages in more generic way based on partitioning metadata
            // and not specifically for this operator
            match agg.mode() {
                AggregateMode::Final => {
                    let mut new_children: Vec<Arc<dyn ExecutionPlan>> = vec![];
                    for child in &children {
                        let new_stage =
                            create_query_stage(job_uuid, self.next_stage_id(), child.clone())?;
                        new_children.push(Arc::new(UnresolvedShuffleExec::new(
                            vec![new_stage.stage_id],
                            new_stage.schema().clone(),
                            new_stage.output_partitioning().partition_count(),
                        )));
                        stages.push(new_stage);
                    }
                    Ok((agg.with_new_children(new_children)?, stages))
                }
                AggregateMode::Partial => Ok((agg.with_new_children(children)?, stages)),
            }
        } else if let Some(join) = execution_plan.as_any().downcast_ref::<HashJoinExec>() {
            Ok((join.with_new_children(children)?, stages))
        } else {
            // TODO check for compatible partitioning schema, not just count
            if execution_plan.output_partitioning().partition_count()
                != children[0].output_partitioning().partition_count()
            {
                let mut new_children: Vec<Arc<dyn ExecutionPlan>> = vec![];
                for child in &children {
                    let new_stage =
                        create_query_stage(job_uuid, self.next_stage_id(), child.clone())?;
                    new_children.push(Arc::new(UnresolvedShuffleExec::new(
                        vec![new_stage.stage_id],
                        new_stage.schema().clone(),
                        new_stage.output_partitioning().partition_count(),
                    )));
                    stages.push(new_stage);
                }
                Ok((execution_plan.with_new_children(new_children)?, stages))
            } else {
                Ok((execution_plan.with_new_children(children)?, stages))
            }
        }
    }

    /// Generate a new stage ID
    fn next_stage_id(&mut self) -> usize {
        self.next_stage_id += 1;
        self.next_stage_id
    }
}

fn execute(
    stages: Vec<Arc<QueryStageExec>>,
    executors: Vec<ExecutorMeta>,
) -> SendableExecutionPlan {
    Box::pin(async move {
        let mut partition_locations: HashMap<usize, Vec<PartitionLocation>> = HashMap::new();
        let mut result_partition_locations = vec![];
        for stage in &stages {
            debug!("execute() {}", &format!("{:?}", stage)[0..60]);
            let stage = remove_unresolved_shuffles(stage.as_ref(), &partition_locations)?;
            let stage = stage.as_any().downcast_ref::<QueryStageExec>().unwrap();
            result_partition_locations = execute_query_stage(
                &stage.job_uuid.clone(),
                stage.stage_id,
                stage.children()[0].clone(),
                executors.clone(),
            )
            .await?;
            partition_locations.insert(stage.stage_id, result_partition_locations.clone());
        }

        let shuffle_reader: Arc<dyn ExecutionPlan> = Arc::new(ShuffleReaderExec::try_new(
            result_partition_locations,
            stages.last().unwrap().schema(),
        )?);
        Ok(shuffle_reader)
    })
}

fn remove_unresolved_shuffles(
    stage: &dyn ExecutionPlan,
    partition_locations: &HashMap<usize, Vec<PartitionLocation>>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let mut new_children: Vec<Arc<dyn ExecutionPlan>> = vec![];
    for child in stage.children() {
        if let Some(unresolved_shuffle) = child.as_any().downcast_ref::<UnresolvedShuffleExec>() {
            let relevant_locations: Vec<_> = unresolved_shuffle
                .query_stage_ids
                .iter()
                .flat_map(|id| partition_locations[id].clone())
                .collect();
            new_children.push(Arc::new(ShuffleReaderExec::try_new(
                relevant_locations,
                unresolved_shuffle.schema().clone(),
            )?))
        } else {
            new_children.push(remove_unresolved_shuffles(
                child.as_ref(),
                partition_locations,
            )?);
        }
    }
    Ok(stage.with_new_children(new_children)?)
}

fn create_query_stage(
    job_uuid: &Uuid,
    stage_id: usize,
    plan: Arc<dyn ExecutionPlan>,
) -> Result<Arc<QueryStageExec>> {
    Ok(Arc::new(QueryStageExec::try_new(
        *job_uuid, stage_id, plan,
    )?))
}

/// Execute a query stage by sending each partition to an executor
async fn execute_query_stage(
    job_uuid: &Uuid,
    stage_id: usize,
    plan: Arc<dyn ExecutionPlan>,
    executors: Vec<ExecutorMeta>,
) -> Result<Vec<PartitionLocation>> {
    info!(
        "execute_query_stage() stage_id={}\n{}",
        stage_id,
        format_plan(plan.as_ref(), 0)?
    );

    let _job_uuid = *job_uuid;
    let partition_count = plan.output_partitioning().partition_count();
    let mut meta = Vec::with_capacity(partition_count);

    let num_chunks = partition_count / executors.len();
    let num_chunks = num_chunks.max(1);
    let partition_chunks: Vec<Vec<usize>> = (0..partition_count)
        .collect::<Vec<usize>>()
        .chunks(num_chunks)
        .map(|r| r.to_vec())
        .collect();

    info!(
        "Executing query stage with {} chunks of partition ranges",
        partition_chunks.len()
    );

    // build metadata for partition locations
    for i in 0..partition_chunks.len() {
        let executor_meta = &executors[i % executors.len()];
        for part in &partition_chunks[i] {
            meta.push(PartitionLocation {
                partition_id: PartitionId::new(_job_uuid, stage_id, *part),
                executor_meta: executor_meta.clone(),
            });
        }
    }

    let mut executions = Vec::with_capacity(partition_count);
    for i in 0..partition_chunks.len() {
        let _plan = plan.clone();
        let executor_meta = executors[i % executors.len()].clone();
        let partition_ids = partition_chunks[i].to_vec();
        executions.push(tokio::spawn(async move {
            let mut client =
                BallistaClient::try_new(&executor_meta.host, executor_meta.port).await?;
            client
                .execute_partition(_job_uuid, stage_id, partition_ids, _plan)
                .await
        }));
    }

    // wait for all partitions to complete
    let results = futures::future::join_all(executions).await;

    // check for errors
    for result in results {
        match result {
            Ok(partition_result) => {
                let final_result = partition_result?;
                debug!("Query stage partition result: {:?}", final_result);
            }
            Err(e) => {
                return Err(BallistaError::General(format!(
                    "Query stage {} failed: {:?}",
                    stage_id, e
                )))
            }
        }
    }

    debug!(
        "execute_query_stage() stage_id={} produced {:?}",
        stage_id, meta
    );

    Ok(meta)
}

pub fn produce_diagram(filename: &str, stages: &[Arc<QueryStageExec>]) -> Result<()> {
    let write_file = File::create(filename)?;
    let mut w = BufWriter::new(&write_file);
    write!(w, "digraph G {{\n")?;

    // draw stages and entities
    for stage in stages {
        write!(w, "\tsubgraph cluster{} {{\n", stage.stage_id)?;
        write!(w, "\t\tlabel = \"Stage {}\";\n", stage.stage_id)?;
        let mut id = AtomicUsize::new(0);
        build_exec_plan_diagram(&mut w, stage.child.as_ref(), stage.stage_id, &mut id, true)?;
        write!(w, "\t}}\n")?;
    }

    // draw relationships
    for stage in stages {
        let mut id = AtomicUsize::new(0);
        build_exec_plan_diagram(&mut w, stage.child.as_ref(), stage.stage_id, &mut id, false)?;
    }

    write!(w, "}}")?;
    Ok(())
}

fn build_exec_plan_diagram(w: &mut BufWriter<&File>,
                           plan: &dyn ExecutionPlan,
                           stage_id: usize,
                           id: &mut AtomicUsize,
                            draw_entity: bool) -> Result<usize> {
    let operator_str = if let Some(exec) = plan.as_any().downcast_ref::<HashAggregateExec>() {
        "HashAggregateExec"
    } else if let Some(exec) = plan.as_any().downcast_ref::<SortExec>() {
        "SortExec"
    } else if let Some(exec) = plan.as_any().downcast_ref::<ProjectionExec>() {
        "ProjectionExec"
    } else if let Some(exec) = plan.as_any().downcast_ref::<HashJoinExec>() {
        "HashJoinExec"
    } else if let Some(exec) = plan.as_any().downcast_ref::<ParquetExec>() {
        "ParquetExec"
    } else if let Some(exec) = plan.as_any().downcast_ref::<CsvExec>() {
        "CsvExec"
    } else if let Some(exec) = plan.as_any().downcast_ref::<FilterExec>() {
        "FilterExec"
    } else if let Some(exec) = plan.as_any().downcast_ref::<QueryStageExec>() {
        "QueryStageExec"
    } else if let Some(exec) = plan.as_any().downcast_ref::<UnresolvedShuffleExec>() {
        "UnresolvedShuffleExec"
    } else if let Some(exec) = plan.as_any().downcast_ref::<CoalesceBatchesExec>() {
        "CoalesceBatchesExec"
    } else if plan.as_any().downcast_ref::<MergeExec>().is_some() {
        "MergeExec"
    } else {
        println!("Unknown: {:?}", plan);
        "Unknown"
    };

    let node_id = id.load(Ordering::SeqCst);
    id.store(node_id + 1, Ordering::SeqCst);

    if draw_entity {
        write!(w, "\t\tstage_{}_exec_{} [shape=box, label=\"{}\"];\n", stage_id, node_id, operator_str)?;
    }
    for child in plan.children() {
        if let Some(x) = child.as_any().downcast_ref::<UnresolvedShuffleExec>() {
            println!("UnresolvedShuffleExec: {:?}", x.query_stage_ids);
            if !draw_entity {
                for y in &x.query_stage_ids {
                    write!(w, "\tstage_{}_exec_{} -> stage_{}_exec_{};\n", y, 1, stage_id, node_id)?;
                }
            }
        } else {
            // relationships within same entity
            let child_id = build_exec_plan_diagram(w, child.as_ref(), stage_id, id, draw_entity)?;
            if draw_entity {
                write!(w, "\t\tstage_{}_exec_{} -> stage_{}_exec_{};\n", stage_id, child_id, stage_id, node_id)?;
            }
        }
    }
    Ok(node_id)
}

#[cfg(test)]
mod test {
    use crate::scheduler::execution_plans::QueryStageExec;
    use crate::scheduler::planner::DistributedPlanner;
    use crate::serde::protobuf;
    use crate::serde::scheduler::ExecutorMeta;
    use crate::test_utils;
    use crate::test_utils::{datafusion_test_context, TPCH_TABLES};
    use crate::utils::format_plan;
    use crate::{error::BallistaError, scheduler::execution_plans::UnresolvedShuffleExec};
    use arrow::datatypes::DataType;
    use datafusion::physical_plan::csv::CsvReadOptions;
    use datafusion::physical_plan::hash_aggregate::HashAggregateExec;
    use datafusion::physical_plan::projection::ProjectionExec;
    use datafusion::physical_plan::sort::SortExec;
    use datafusion::physical_plan::ExecutionPlan;
    use datafusion::prelude::*;
    use datafusion::{execution::context::ExecutionContext, physical_plan::merge::MergeExec};
    use std::convert::TryInto;
    use std::sync::Arc;
    use uuid::Uuid;

    macro_rules! downcast_exec {
        ($exec: expr, $ty: ty) => {
            $exec.as_any().downcast_ref::<$ty>().unwrap()
        };
    }

    #[test]
    fn test() -> Result<(), BallistaError> {
        let mut ctx = datafusion_test_context("testdata")?;

        // simplified form of TPC-H query 1
        let df = ctx.sql(
            "select l_returnflag, sum(l_extendedprice * 1) as sum_disc_price
            from lineitem
            group by l_returnflag
            order by l_returnflag",
        )?;

        let plan = df.to_logical_plan();
        let plan = ctx.optimize(&plan)?;
        let plan = ctx.create_physical_plan(&plan)?;

        let mut planner = DistributedPlanner::try_new(vec![ExecutorMeta {
            id: "".to_string(),
            host: "".to_string(),
            port: 0,
        }])?;
        let job_uuid = Uuid::new_v4();
        let stages = planner.plan_query_stages(&job_uuid, plan)?;
        for stage in &stages {
            println!("{}", format_plan(stage.as_ref(), 0)?);
        }

        /* Expected result:
        QueryStageExec: job=f011432e-e424-4016-915d-e3d8b84f6dbd, stage=1
         HashAggregateExec: groupBy=["l_returnflag"], aggrExpr=["SUM(l_extendedprice Multiply Int64(1)) [\"l_extendedprice * CAST(1 AS Float64)\"]"]
          CsvExec: testdata/lineitem; partitions=2

        QueryStageExec: job=f011432e-e424-4016-915d-e3d8b84f6dbd, stage=2
         MergeExec
          UnresolvedShuffleExec: stages=[1]

        QueryStageExec: job=f011432e-e424-4016-915d-e3d8b84f6dbd, stage=3
         SortExec { input: ProjectionExec { expr: [(Column { name: "l_returnflag" }, "l_returnflag"), (Column { name: "SUM(l_ext
          ProjectionExec { expr: [(Column { name: "l_returnflag" }, "l_returnflag"), (Column { name: "SUM(l_extendedprice Multip
           HashAggregateExec: groupBy=["l_returnflag"], aggrExpr=["SUM(l_extendedprice Multiply Int64(1)) [\"l_extendedprice * CAST(1 AS Float64)\"]"]
            UnresolvedShuffleExec: stages=[2]
        */

        let sort = stages[2].children()[0].clone();
        let sort = downcast_exec!(sort, SortExec);

        let projection = sort.children()[0].clone();
        println!("{:?}", projection);
        let projection = downcast_exec!(projection, ProjectionExec);

        let final_hash = projection.children()[0].clone();
        let final_hash = downcast_exec!(final_hash, HashAggregateExec);

        let unresolved_shuffle = final_hash.children()[0].clone();
        let unresolved_shuffle = downcast_exec!(unresolved_shuffle, UnresolvedShuffleExec);
        assert_eq!(unresolved_shuffle.query_stage_ids, vec![2]);

        let merge_exec = stages[1].children()[0].clone();
        let merge_exec = downcast_exec!(merge_exec, MergeExec);

        let unresolved_shuffle = merge_exec.children()[0].clone();
        let unresolved_shuffle = downcast_exec!(unresolved_shuffle, UnresolvedShuffleExec);
        assert_eq!(unresolved_shuffle.query_stage_ids, vec![1]);

        let partial_hash = stages[0].children()[0].clone();
        let partial_hash_serde = roundtrip_operator(partial_hash.clone())?;

        let partial_hash = downcast_exec!(partial_hash, HashAggregateExec);
        let partial_hash_serde = downcast_exec!(partial_hash_serde, HashAggregateExec);

        assert_eq!(
            format!("{:?}", partial_hash),
            format!("{:?}", partial_hash_serde)
        );

        Ok(())
    }

    fn roundtrip_operator(
        plan: Arc<dyn ExecutionPlan>,
    ) -> Result<Arc<dyn ExecutionPlan>, BallistaError> {
        let proto: protobuf::PhysicalPlanNode = plan.clone().try_into()?;
        let result_exec_plan: Arc<dyn ExecutionPlan> = (&proto).try_into()?;
        Ok(result_exec_plan)
    }
}
