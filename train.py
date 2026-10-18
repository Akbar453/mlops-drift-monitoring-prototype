import json
from src.core import train

if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
