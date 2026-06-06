from fastapi import APIRouter
from sqlalchemy import text

from ..dependencies import DatabaseDep, SettingsDep
from ..schemas.api.health import HealthResponse, ServiceStatus

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(settings: SettingsDep, database: DatabaseDep) -> HealthResponse:
    """Week 1 health check: verifies the database connection."""
    services = {}
    overall_status = "ok"

    try:
        with database.get_session() as session:
            session.execute(text("SELECT 1"))
        services["database"] = ServiceStatus(status="healthy", message="Connected successfully")
    except Exception as e:
        services["database"] = ServiceStatus(status="unhealthy", message=str(e))
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        environment=settings.environment,
        service_name=settings.service_name,
        services=services,
    )