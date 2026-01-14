import os
import datetime as dt
from typing import Optional

from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Text,
    JSON,
    create_engine,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship


DB_PATH = os.environ.get("FLOWMATIC_DB", os.path.join(os.getcwd(), "flowmatic.db"))
ENGINE = create_engine(f"sqlite:///{DB_PATH}", future=True)
SessionLocal = sessionmaker(bind=ENGINE, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    source = Column(String, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    versions = relationship("DatasetVersion", back_populates="dataset", cascade="all, delete-orphan")


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    version = Column(String, nullable=False)
    path = Column(String, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    dataset = relationship("Dataset", back_populates="versions")


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"
    id = Column(Integer, primary_key=True)
    run_id = Column(String, unique=True, nullable=False)
    dataset_version_id = Column(Integer, ForeignKey("dataset_versions.id"), nullable=False)
    pipeline_name = Column(String, nullable=False)
    status = Column(String, default="pending")  # pending|running|succeeded|failed
    params = Column(JSON)
    metrics = Column(JSON)
    artifacts_dir = Column(String)
    error = Column(Text)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    finished_at = Column(DateTime)


class ModelArtifact(Base):
    __tablename__ = "model_artifacts"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    task = Column(String, nullable=False)  # anomaly-detection|forecasting|classification
    dataset_version_id = Column(Integer, ForeignKey("dataset_versions.id"))
    run_id = Column(String)
    path = Column(String, nullable=False)
    metrics = Column(JSON)
    created_at = Column(DateTime, default=dt.datetime.utcnow)


def init_db() -> None:
    Base.metadata.create_all(ENGINE)


def register_dataset(name: str, source: str) -> int:
    with SessionLocal() as s:
        ds = s.query(Dataset).filter_by(name=name).one_or_none()
        if not ds:
            ds = Dataset(name=name, source=source)
            s.add(ds)
            s.commit()
            s.refresh(ds)
        return ds.id


def add_dataset_version(dataset_id: int, version: str, path: str) -> int:
    with SessionLocal() as s:
        dv = DatasetVersion(dataset_id=dataset_id, version=version, path=path)
        s.add(dv)
        s.commit()
        s.refresh(dv)
        return dv.id


def create_run(run_id: str, dataset_version_id: int, pipeline_name: str, params: Optional[dict], artifacts_dir: str) -> None:
    with SessionLocal() as s:
        pr = PipelineRun(
            run_id=run_id,
            dataset_version_id=dataset_version_id,
            pipeline_name=pipeline_name,
            status="running",
            params=params or {},
            artifacts_dir=artifacts_dir,
        )
        s.add(pr)
        s.commit()


def finalize_run(run_id: str, status: str, metrics: Optional[dict] = None, error: Optional[str] = None) -> None:
    with SessionLocal() as s:
        pr = s.query(PipelineRun).filter_by(run_id=run_id).one()
        pr.status = status
        pr.metrics = metrics or {}
        pr.error = error
        pr.finished_at = dt.datetime.utcnow()
        s.commit()


def save_model_artifact(name: str, task: str, path: str, dataset_version_id: Optional[int] = None, run_id: Optional[str] = None, metrics: Optional[dict] = None) -> int:
    with SessionLocal() as s:
        ma = ModelArtifact(
            name=name,
            task=task,
            dataset_version_id=dataset_version_id,
            run_id=run_id,
            path=path,
            metrics=metrics or {},
        )
        s.add(ma)
        s.commit()
        s.refresh(ma)
        return ma.id
