from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

class Neo4jSchemaExplorer:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_node_labels(self):
        query = "CALL db.labels()"
        with self.driver.session() as session:
            return [record["label"] for record in session.run(query)]

    def get_relationship_types(self):
        query = "CALL db.relationshipTypes()"
        with self.driver.session() as session:
            return [record["relationshipType"] for record in session.run(query)]

    def get_node_properties(self):
        query = """
        CALL db.schema.nodeTypeProperties() 
        YIELD nodeLabels, propertyName, propertyTypes
        RETURN nodeLabels, propertyName, propertyTypes
        ORDER BY nodeLabels
        """
        with self.driver.session() as session:
            return list(session.run(query))

    def get_relationship_properties(self):
        query = """
        CALL db.schema.relTypeProperties()
        YIELD relType, propertyName, propertyTypes
        RETURN relType, propertyName, propertyTypes
        ORDER BY relType
        """
        with self.driver.session() as session:
            return list(session.run(query))


if __name__ == "__main__":
    explorer = Neo4jSchemaExplorer(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    print("🔹 Node Labels:")
    print(explorer.get_node_labels())

    print("\n🔹 Relationship Types:")
    print(explorer.get_relationship_types())

    print("\n🔹 Node Properties:")
    for record in explorer.get_node_properties():
        print(f"{record['nodeLabels']} → {record['propertyName']} ({record['propertyTypes']})")

    print("\n🔹 Relationship Properties:")
    for record in explorer.get_relationship_properties():
        print(f"{record['relType']} → {record['propertyName']} ({record['propertyTypes']})")

    explorer.close()
