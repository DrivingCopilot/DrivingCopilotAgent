import os
import asyncio
import logging
from typing import TypedDict, Any, List, Dict, Optional

from tenacity import retry, stop_after_attempt, wait_exponential
from neo4j import AsyncGraphDatabase
from neo4j_graphrag.experimental.components.schema import SchemaBuilder
from neo4j_graphrag.experimental.pipeline import Pipeline
from neo4j_graphrag.experimental.components.entity_relation_extractor import LLMEntityRelationExtractor
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from app.agent.state import AgentState

from app.graph.schema import DrivingGraphSchema

logger = logging.getLogger(__name__)

# --- 1. Abstracted Local LLM Interface ---
class LocalQwen2VL(BaseChatModel):
    model_name: str = "qwen2-vl-7b-instruct-int4"
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("Requires local TensorRT-LLM binding implementation")
        
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        # Implementation for invoking local TensorRT-LLM asynchronously goes here
        raise NotImplementedError("Requires local TensorRT-LLM binding implementation")
        
    @property
    def _llm_type(self) -> str:
        return "local-qwen2-vl-tensorrt"


# --- 3. MCP Tool Interface (Input/Output Schemas) ---
class ExtractionInput(BaseModel):
    text: str = Field(..., description="Vehicle diagnostic text (DTC, symptoms) to extract entities from.")

class ExtractionOutput(BaseModel):
    success: bool
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    context: str
    error: Optional[str] = None

# --- Main Graph RAG Component ---
class VehicleGraphManager:
    """
    Manages Neo4j Knowledge Graph operations for the Vehicle Copilot.
    Can be used as a LangGraph Node ('Knowledge Agent') or registered as an MCP Tool.
    """
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        
        # Async Driver setup for non-blocking I/O
        self.driver = AsyncGraphDatabase.driver(
            self.uri, 
            auth=(self.user, self.password)
        )
        
        # Build Schema Pipeline
        self.schema_builder = SchemaBuilder(
            node_types=DrivingGraphSchema.get_node_types(),
            relationship_types=DrivingGraphSchema.get_relationship_types(),
            patterns=DrivingGraphSchema.get_patterns()
        )
        self.pipe = Pipeline()
        self.pipe.add_component(self.schema_builder, "schema_builder")
        
        # LLM Extractor using the abstracted Local LLM
        self.extractor = LLMEntityRelationExtractor(
            llm=LocalQwen2VL(),
            prompt_template="Extract entities and relationships from the following vehicle diagnostic text based on the schema.",
            create_lexical_graph=True
        )

    async def close(self):
        """Close the async Neo4j driver connection."""
        await self.driver.close()

    @retry(
        stop=stop_after_attempt(3), # Initial try + 2 retries = 3 attempts total
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def extracting_data(self, text: str) -> dict:
        """
        Async extraction of entities and relationships, and storage to Neo4j.
        Includes failure handling with tenacity (max 2 retries).
        """
        try:
            extraction_result = await asyncio.to_thread(self.extractor.run, text)

            return {
                "success": True,
                "data": getattr(extraction_result, "dict", lambda: extraction_result)()
            }
        except Exception as e:
            logger.error(f"Failed to extract and store graph data: {e}")
            raise # Triggers Tenacity retry mechanism

    def _to_natural_language(self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> str:
        """
        Converts Graph Traversal results into 'Natural Language Context' 
        for easy context fusion with Vector RAG results.
        """
        if not entities and not relationships:
            return "No relevant vehicle graph context found."

        sentences = []
        for rel in relationships:
            source = rel.get("source", "Unknown")
            target = rel.get("target", "Unknown")
            rel_type = rel.get("type", "RELATES_TO")
            # Format relation type (e.g., 'HAS_PART' -> 'has part')
            formatted_type = str(rel_type).lower().replace("_", " ")
            sentences.append(f"The {source} {formatted_type} the {target}.")
            
        for entity in entities:
            # Example fallback if relationships aren't dense enough
            label = entity.get("label", "Entity")
            props = entity.get("properties", {})
            name = props.get("name", entity.get("id", "Unknown"))
            sentences.append(f"Found {label}: {name}.")
            
        return " ".join(sentences)


    async def mcp_run_extraction(self, input_data: ExtractionInput) -> ExtractionOutput:
        """
        MCP Tool wrapper: Exposes extraction logic as an MCP-compatible interface.
        """
        try:
            result = await self.extracting_data(input_data.text)
            ext_data = result.get("data", {})
            

            entities = ext_data.get("entities", []) if isinstance(ext_data, dict) else []
            relationships = ext_data.get("relationships", []) if isinstance(ext_data, dict) else []
            
            nl_context = self._to_natural_language(entities, relationships)
            

            # JSON 형식으로 데이터 반환
            return ExtractionOutput(
                success=True,
                entities=entities,
                relationships=relationships,
                context=nl_context
            )
        except Exception as e:
            return ExtractionOutput(
                success=False,
                entities=[],
                relationships=[],
                context="",
                error=str(e)
            )

