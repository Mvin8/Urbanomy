import uuid

def get_id(n : int = 6):
    uuid4 = uuid.uuid4().hex[:n]
    return str(uuid4).replace('-', '')

__all__=[
    'get_id'
]