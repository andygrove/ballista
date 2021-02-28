// Copyright 2021 Andy Grove
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

use std::sync::Arc;
use std::{any::Any, pin::Pin};

use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::physical_plan::{ExecutionPlan, Partitioning};
use datafusion::{error::Result, physical_plan::RecordBatchStream};
use uuid::Uuid;
use crate::scheduler::execution_plans::{ShuffleReaderExec, UnresolvedShuffleExec};
use std::collections::HashSet;

/// QueryStageExec represents a section of a query plan that has consistent partitioning and
/// can be executed as one unit with each partition being executed in parallel. The output of
/// a query stage either forms the input of another query stage or can be the final result of
/// a query.
#[derive(Debug, Clone)]
pub struct QueryStageExec {
    /// Unique ID for the job (query) that this stage is a part of
    pub(crate) job_uuid: Uuid,
    /// Unique query stage ID within the job
    pub(crate) stage_id: usize,
    /// Physical execution plan for this query stage
    pub(crate) child: Arc<dyn ExecutionPlan>,
}

impl QueryStageExec {
    /// Create a new query stage
    pub fn try_new(job_uuid: Uuid, stage_id: usize, child: Arc<dyn ExecutionPlan>) -> Result<Self> {
        Ok(Self {
            job_uuid,
            stage_id,
            child,
        })
    }

    /// Get a list of query stages that this query stage depends on as direct children
    pub fn get_child_stages(&self) -> HashSet<usize> {
        let mut accum = HashSet::new();
        find_child_stages(&self.child, &mut accum);
        accum
    }
}

fn find_child_stages(plan: &Arc<dyn ExecutionPlan>, accum: &mut HashSet<usize>) {
    if let Some(shuffle_reader) = plan.as_any().downcast_ref::<ShuffleReaderExec>() {
        shuffle_reader.partition_location.iter()
            .for_each(|loc| {
                accum.insert(loc.partition_id.stage_id);
            });
    } else if let Some(unresolved_shuffle_exec) = plan.as_any().downcast_ref::<UnresolvedShuffleExec>() {
        for id in &unresolved_shuffle_exec.query_stage_ids {
            accum.insert(*id);
        }
    }
    plan.children().iter().for_each(|child| find_child_stages(&child, accum));
}

#[async_trait]
impl ExecutionPlan for QueryStageExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.child.schema()
    }

    fn output_partitioning(&self) -> Partitioning {
        self.child.output_partitioning()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.child.clone()]
    }

    fn with_new_children(
        &self,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        assert!(children.len() == 1);
        Ok(Arc::new(QueryStageExec::try_new(
            self.job_uuid,
            self.stage_id,
            children[0].clone(),
        )?))
    }

    async fn execute(
        &self,
        partition: usize,
    ) -> Result<Pin<Box<dyn RecordBatchStream + Send + Sync>>> {
        self.child.execute(partition).await
    }
}
