from neo4j_graphrag.experimental.components.schema import (
    NodeType,
    PropertyType,
    RelationshipType,
)

class DrivingGraphSchema:
    """
    Schema definition for the Vehicle Diagnostics Knowledge Graph.
    Separated for maintainability and configuration management.
    """
    
    @staticmethod
    def get_node_types() -> list[NodeType]:
        return [
            NodeType(label="Component", description="Vehicle components",
                properties=[
                    PropertyType(name="name", type="STRING", description="Name of the component"),
                    PropertyType(name="category", type="STRING", description="Diagnostic category (e.g., engine, transmission, brakes)"),
                ]        
            ),
            NodeType(label="WarningLight", description="Vehicle warning lights",
                properties=[
                    PropertyType(name="name", type="STRING", description="Name of the warning light"),
                    PropertyType(name="color", type="STRING", description="Color of the warning light"),
                ]
            ),
            NodeType(label="Symptom", description="Vehicle symptoms",
                properties=[
                    PropertyType(name="description", type="STRING", description="Description of the symptom"),
                    PropertyType(name="severity", type="INTEGER", description="Severity level of the symptom (e.g., 1-10)"),
                ]
            ),
            NodeType(label="Maintenance", description="Maintenance actions",
                properties=[
                    PropertyType(name="type", type="STRING", description="Type of maintenance (e.g., oil change, brake inspection)"),
                    PropertyType(name="interval_months", type="INTEGER", description="Maintenance interval in months"),
                    PropertyType(name="interval_miles", type="INTEGER", description="Maintenance interval in miles"),
                ]
            ),
            NodeType(label="DTC Code", description="Diagnostic Trouble Codes",
                properties=[
                    PropertyType(name="code", type="STRING", description="DTC code (e.g., P0300)"),
                    PropertyType(name="description", type="STRING", description="Description of the DTC code"),
                ]
            ),
            NodeType(label="System", description="Vehicle systems",
                properties=[
                    PropertyType(name="name", type="STRING", description="Name of the vehicle system (e.g., engine, transmission)"),
                ]
            ),
            NodeType(label="Action", description="Recommended actions",
                properties=[
                    PropertyType(name="description", type="STRING", description="Description of the recommended action"),
                    PropertyType(name="urgency", type="STRING", description="Urgency level of the action (e.g., low, medium, high)"),
                ]
            ),
            NodeType(label="Schedule", description="Maintenance schedules",
                properties=[
                    PropertyType(name="value", type="INTEGER", description="Value representing the maintenance schedule"),
                    PropertyType(name="unit", type="STRING", description="Unit of the maintenance schedule"),
                ]
            )
        ]

    @staticmethod
    def get_relationship_types() -> list[RelationshipType]:
        return [
            RelationshipType(label="HAS_PART", description="Component relationships"),
            RelationshipType(label="MAINTAINED_BY", description="Maintenance relationships"),
            RelationshipType(label="INDICATES", description="Warning light indicates symptom"),
            RelationshipType(label="CAUSED_BY", description="WarningLight caused by component"),
            RelationshipType(label="SYMPTOM_OF", description="Symptom of component"),
            RelationshipType(label="RESOLVED_BY", description="Symptom resolved by action"),
            RelationshipType(label="APPLIES_TO", description="Maintenance applies to component"),
            RelationshipType(label="HAS_INTERVAL", description="Maintenance has interval"),
            RelationshipType(label="TRIGGERS", description="DTC Code triggers warning light"),
            RelationshipType(label="MAPS_TO", description="DTC Code maps to component"),
        ]

    @staticmethod
    def get_patterns() -> list[tuple[str, str, str]]:
        return [
            ("Component", "HAS_PART", "System"),
            ("Component", "MAINTAINED_BY", "Maintenance"),
            ("WarningLight", "INDICATES", "Symptom"),
            ("WarningLight", "CAUSED_BY", "Component"),
            ("Symptom", "SYMPTOM_OF", "Component"),
            ("Symptom", "RESOLVED_BY", "Action"),
            ("Maintenance", "APPLIES_TO", "Component"),
            ("Maintenance", "HAS_INTERVAL", "Schedule"),
            ("DTC Code", "MAPS_TO", "Component"),
            ("DTC Code", "TRIGGERS", "WarningLight"),
        ]
