"""Admin API routes for tests and database exploration."""

import subprocess
import json
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text, inspect

from webapp.database.connection import get_db, engine
from webapp.database.models import User
from webapp.auth.dependencies import get_current_user

router = APIRouter()


@router.get("/tests/run")
async def run_tests(current_user: User = Depends(get_current_user)):
    """Run pytest and return results."""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/", "-v", "--tb=short", "--json-report", "--json-report-file=-"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd="C:/Users/luado/Desktop/Claude_Projects/Analisis_Imagenes"
        )

        # Parse JSON output if available
        lines = result.stdout.split('\n')
        json_line = None
        for line in lines:
            if line.startswith('{') and '"created"' in line:
                json_line = line
                break

        if json_line:
            report = json.loads(json_line)
            return {
                "success": True,
                "summary": report.get("summary", {}),
                "tests": report.get("tests", []),
                "duration": report.get("duration", 0)
            }
        else:
            # Fallback: parse text output
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr
            }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Tests timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/tests/results")
async def get_test_results(current_user: User = Depends(get_current_user)):
    """Get last test results from simple pytest run."""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/", "-v", "--tb=line"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd="C:/Users/luado/Desktop/Claude_Projects/Analisis_Imagenes"
        )

        # Parse output
        lines = result.stdout.split('\n')
        tests = []
        summary = {}

        for line in lines:
            if '::' in line and ('PASSED' in line or 'FAILED' in line or 'ERROR' in line):
                parts = line.strip().split(' ')
                test_name = parts[0]
                status = 'passed' if 'PASSED' in line else ('failed' if 'FAILED' in line else 'error')
                tests.append({
                    "name": test_name,
                    "status": status
                })
            elif 'passed' in line and ('warning' in line or 'second' in line):
                summary["line"] = line.strip()

        return {
            "success": result.returncode == 0,
            "tests": tests,
            "total": len(tests),
            "passed": sum(1 for t in tests if t["status"] == "passed"),
            "failed": sum(1 for t in tests if t["status"] == "failed"),
            "summary": summary.get("line", ""),
            "return_code": result.returncode
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/database/tables")
async def get_tables(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get list of all database tables."""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    inspector = inspect(engine)
    tables = []

    for table_name in inspector.get_table_names():
        columns = inspector.get_columns(table_name)
        # Count rows
        count_result = db.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        row_count = count_result.scalar()

        tables.append({
            "name": table_name,
            "columns": [{"name": c["name"], "type": str(c["type"])} for c in columns],
            "row_count": row_count
        })

    return {"tables": tables}


@router.get("/database/tables/{table_name}")
async def get_table_data(
    table_name: str,
    limit: int = 100,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get data from a specific table."""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    # Validate table name exists
    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        raise HTTPException(status_code=404, detail="Table not found")

    # Get columns
    columns = [c["name"] for c in inspector.get_columns(table_name)]

    # Get data
    result = db.execute(text(f"SELECT * FROM {table_name} LIMIT {limit} OFFSET {offset}"))
    rows = []
    for row in result:
        row_dict = {}
        for i, col in enumerate(columns):
            value = row[i]
            # Handle non-JSON-serializable types
            if isinstance(value, bytes):
                value = value.hex()
            elif hasattr(value, 'isoformat'):
                value = value.isoformat()
            row_dict[col] = value
        rows.append(row_dict)

    # Get total count
    count_result = db.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
    total = count_result.scalar()

    return {
        "table": table_name,
        "columns": columns,
        "rows": rows,
        "total": total,
        "limit": limit,
        "offset": offset
    }
