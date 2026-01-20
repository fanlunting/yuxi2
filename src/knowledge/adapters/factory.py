import os

from src.knowledge.utils.kb_utils import derive_kb_node_label

from .base import GraphAdapter
from .lightrag import LightRAGGraphAdapter
from .upload import UploadGraphAdapter


class GraphAdapterFactory:
    """图谱适配器工厂 (Graph Adapter Factory)"""

    _registry: dict[str, type[GraphAdapter]] = {
        "upload": UploadGraphAdapter,
        "lightrag": LightRAGGraphAdapter,
    }

    @classmethod
    def register(cls, graph_type: str, adapter_class: type[GraphAdapter]):
        """注册适配器类 (Register adapter class)"""
        cls._registry[graph_type] = adapter_class

    @classmethod
    def create_adapter(cls, graph_type: str, **kwargs) -> GraphAdapter:
        """创建适配器实例 (Create adapter instance)"""
        adapter_class = cls._registry.get(graph_type)
        if not adapter_class:
            raise ValueError(f"Unknown graph type: {graph_type}")

        return adapter_class(**kwargs)

    @classmethod
    def get_supported_types(cls) -> dict[str, str]:
        """获取支持的图谱类型及其描述"""
        return {
            "upload": "上传文件图谱 - 支持embedding和阈值查询",
            "lightrag": "LightRAG知识图谱 - 基于kb_id标签的图谱",
        }

    @classmethod
    def detect_graph_type(cls, db_id: str, knowledge_base_manager=None) -> str:
        """
        自动检测图谱类型

        Args:
            db_id: 数据库ID
            knowledge_base_manager: 知识库管理器实例

        Returns:
            图谱类型: "lightrag" (LightRAG) 或 "upload"
        """
        # 1. 仅当知识库类型本身是 lightrag 时，才视为 lightrag 图谱
        if knowledge_base_manager and hasattr(knowledge_base_manager, "is_lightrag_database"):
            if knowledge_base_manager.is_lightrag_database(db_id):
                return "lightrag"

        # 2. 其他情况默认为 Upload 类型（包括 milvus 等）
        return "upload"

    @classmethod
    def create_adapter_by_db_id(cls, db_id: str, knowledge_base_manager=None, graph_db_instance=None) -> GraphAdapter:
        """
        根据数据库ID自动创建对应的适配器

        Args:
            db_id: 数据库ID
            knowledge_base_manager: 知识库管理器实例
            graph_db_instance: 图数据库实例 (用于Upload类型)

        Returns:
            对应的图谱适配器
        """
        graph_type = cls.detect_graph_type(db_id, knowledge_base_manager)

        if graph_type == "lightrag":
            # LightRAG 类型，使用 kb_id 作为配置
            return cls.create_adapter("lightrag", config={"kb_id": db_id})
        else:
            # Upload 类型：Neo4j Community 默认单库，用 kb_label 做隔离
            return cls.create_adapter(
                "upload",
                graph_db_instance=graph_db_instance,
                config={
                    "kgdb_name": os.environ.get("NEO4J_DATABASE", "neo4j"),
                    "kb_label": derive_kb_node_label(db_id),
                    "kb_id": db_id,
                },
            )

    @classmethod
    def create_adapter_for_db_id(cls, db_id: str, knowledge_base_manager=None, graph_db_instance=None) -> GraphAdapter:
        """
        兼容性方法，调用 create_adapter_by_db_id
        """
        return cls.create_adapter_by_db_id(db_id, knowledge_base_manager, graph_db_instance)
