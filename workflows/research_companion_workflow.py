from typing import Dict, Any, List, Optional
from agents.research_coordinator import research_coordinator

class ResearchCompanionWorkflow:
    name: str = "research_companion_workflow"

    def execute(
        self,
        user_request: str,
        uploaded_files: Optional[List[Any]] = None,
        file_names: Optional[List[str]] = None,
        use_watsonx_live: bool = False
    ) -> Dict[str, Any]:
        ingestion_status = None
        if uploaded_files:
            ingestion_output = research_coordinator.handle_request(
                user_message=user_request or "Ingest uploaded documents",
                uploaded_files=uploaded_files,
                file_names=file_names,
                use_watsonx_live=use_watsonx_live
            )
            ingestion_status = ingestion_output

        if user_request:
            query_output = research_coordinator.handle_request(
                user_message=user_request,
                uploaded_files=None,
                use_watsonx_live=use_watsonx_live
            )
            return {
                "workflow": self.name,
                "status": "success",
                "ingestion_result": ingestion_status,
                "result": query_output,
                "final_response": query_output["response"]
            }
        
        return {
            "workflow": self.name,
            "status": "success",
            "ingestion_result": ingestion_status,
            "result": ingestion_status,
            "final_response": ingestion_status["response"] if ingestion_status else "No request provided."
        }

research_companion_workflow = ResearchCompanionWorkflow()
