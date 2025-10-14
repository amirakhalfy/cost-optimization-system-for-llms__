from fastapi import FastAPI
from api.sql_router import router as sql_router
from api.apiaggregation import router as aggregation_router
from api.advanced_comparison import router as advanced_comparison_router
from api.optimization_router import router as optimization_router
app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "healthy"}
app.include_router(sql_router)
app.include_router(aggregation_router)
app.include_router(advanced_comparison_router)
app.include_router(optimization_router)