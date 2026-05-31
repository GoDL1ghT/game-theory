"""Pydantic-схемы валидации входных параметров ServerBalancer."""
from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field, field_validator, model_validator


class ServerSpec(BaseModel):
    """Описание одного сервера в пуле."""

    name: str = Field(..., description="Идентификатор сервера")
    mu: float = Field(..., gt=0, description="Интенсивность обслуживания, запр./с")


class Scenario(BaseModel):
    """Полный сценарий балансировки, загружаемый из scenario.yaml."""

    servers: List[ServerSpec] = Field(..., min_length=1)
    total_lambda: float = Field(..., ge=0, description="Суммарный поток, запр./с")
    rho_max: float = Field(0.95, gt=0, le=1.0, description="Предел загрузки сервера")
    sla_w_ms: float = Field(200.0, gt=0, description="SLA по времени отклика, мс")
    sla_p_wait: float = Field(0.5, ge=0, le=1.0, description="SLA по P(ожидание)")

    @field_validator("servers")
    @classmethod
    def _unique_names(cls, v: List[ServerSpec]) -> List[ServerSpec]:
        names = [s.name for s in v]
        if len(names) != len(set(names)):
            raise ValueError("Имена серверов должны быть уникальны")
        return v

    @model_validator(mode="after")
    def _capacity_warning(self) -> "Scenario":
        """Поток не может превышать суммарную мощность пула с учётом rho_max."""
        capacity = self.rho_max * sum(s.mu for s in self.servers)
        if self.total_lambda > capacity:
            raise ValueError(
                f"Поток {self.total_lambda} превышает допустимую мощность "
                f"{capacity:.2f} (rho_max={self.rho_max}). Система перегружена."
            )
        return self

    @property
    def mus(self) -> List[float]:
        return [s.mu for s in self.servers]

    @property
    def names(self) -> List[str]:
        return [s.name for s in self.servers]
