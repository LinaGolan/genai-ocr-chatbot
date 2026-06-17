from __future__ import annotations

"""
Data models for BL283 field extraction.

Two groups, both kept here so the logic modules stay free of type definitions:
  - The Pydantic extraction schema (`FormExtraction` + nested) — the single
    source of truth for the output JSON shape. extractor.py derives the
    Structured Outputs json_schema from it; validator.py verifies the shape.
  - The validation result types (`FieldStatus`, `ValidationResult`) produced by
    validator.validate() and consumed by the frontend.
"""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel


class DateField(BaseModel):
    day: str = ""
    month: str = ""
    year: str = ""


class AddressField(BaseModel):
    street: str = ""
    houseNumber: str = ""
    entrance: str = ""
    apartment: str = ""
    city: str = ""
    postalCode: str = ""
    poBox: str = ""


class MedicalInstitutionFields(BaseModel):
    healthFundMember: str = ""
    natureOfAccident: str = ""
    medicalDiagnoses: str = ""


class FormExtraction(BaseModel):
    lastName: str = ""
    firstName: str = ""
    idNumber: str = ""
    gender: str = ""
    dateOfBirth: DateField = DateField()
    address: AddressField = AddressField()
    landlinePhone: str = ""
    mobilePhone: str = ""
    jobType: str = ""
    dateOfInjury: DateField = DateField()
    timeOfInjury: str = ""
    accidentLocation: str = ""
    accidentAddress: str = ""
    accidentDescription: str = ""
    injuredBodyPart: str = ""
    signature: str = ""
    formFillingDate: DateField = DateField()
    formReceiptDateAtClinic: DateField = DateField()
    medicalInstitutionFields: MedicalInstitutionFields = MedicalInstitutionFields()


# ---------------------------------------------------------------------------
# Validation result types — produced by validator.validate()
# ---------------------------------------------------------------------------

Status = Literal["ok", "uncertain", "invalid"]


@dataclass
class FieldStatus:
    status: Status = "ok"
    reason: str = ""


@dataclass
class ValidationResult:
    fields: dict[str, FieldStatus]
    completeness: float
    accuracy_estimate: Literal["high", "medium", "low"]

    def to_dict(self) -> dict:
        return {
            "fields": {k: {"status": v.status, "reason": v.reason} for k, v in self.fields.items()},
            "completeness": self.completeness,
            "accuracy_estimate": self.accuracy_estimate,
        }
