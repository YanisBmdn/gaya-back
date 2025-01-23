import json

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class Endpoint:
    """Represents an API endpoint with its configuration"""
    url: str
    description: str
    parameters: Optional[Dict[str, str]] = None

    def __str__(self):
        return f"{self.url}: {self.description} \n Parameters: {self.parameters}"


class API():
    """Represents an API with its endpoints"""
    name: str
    endpoints: List[Endpoint]

    def __init__(self, name: str):
        self.name = name
        self.endpoints = []

        if name == "OpenMeteo":
            with open('known_apis.json', 'r') as file:
                parameters = json.load(file)
            
            for endpoint in parameters:
                self.endpoints.append(Endpoint(
                    url=endpoint['url'],
                    description=endpoint['description'],
                    parameters=endpoint['parameters']
                ))

    def __str__(self):
        endpoint_str = "\n".join([str(endpoint) for endpoint in self.endpoints])
        return f"{self.name} API \n Endpoints: {endpoint_str}"

OpenMeteoAPI = API("OpenMeteo")