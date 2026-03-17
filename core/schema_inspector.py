from sqlalchemy import inspect
from core.database import engine

def get_schema_metadata() -> str:
    """
    Uses SQLAlchemy inspector to dynamically fetch database schema.
    Returns a formatted string representing tables and columns.
    """
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    schema_lines = []
    
    for table_name in tables:
        columns = inspector.get_columns(table_name)
        fks = inspector.get_foreign_keys(table_name)
        
        col_details = []
        for col in columns:
            col_name = col['name']
            col_type = col['type']
            col_details.append(f"{col_name} ({col_type})")
            
        pk_constraint = inspector.get_pk_constraint(table_name)
        pks = pk_constraint.get('constrained_columns', []) if pk_constraint else []
        
        schema_lines.append(f"Table: {table_name}")
        schema_lines.append(f"  Columns: {', '.join(col_details)}")
        if pks:
            schema_lines.append(f"  Primary Keys: {', '.join(pks)}")
            
        if fks:
            for fk in fks:
                constrained_cols = ", ".join(fk['constrained_columns'])
                referred_table = fk['referred_table']
                referred_cols = ", ".join(fk['referred_columns'])
                schema_lines.append(f"  Foreign Key: {constrained_cols} -> {referred_table}.{referred_cols}")
                
        schema_lines.append("") # Empty line for spacing
        
    if not schema_lines:
        return "Database is currently empty with no tables."
        
    return "\n".join(schema_lines)
