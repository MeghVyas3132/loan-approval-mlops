"""The data contract for a loan application.

One declaration drives three consumers: the pipeline's validation stage, the
unit tests, and the API request model - so a schema change can never silently
diverge between training and serving.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str                                   # "numeric" | "categorical"
    required: bool = True
    allowed_values: tuple[str, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    nullable: bool = False
    description: str = ""

    def is_numeric(self) -> bool:
        return self.dtype == "numeric"


@dataclass(frozen=True)
class Schema:
    columns: tuple[ColumnSpec, ...] = field(default_factory=tuple)

    def names(self) -> list[str]:
        return [c.name for c in self.columns]

    def numeric(self) -> list[str]:
        return [c.name for c in self.columns if c.is_numeric()]

    def categorical(self) -> list[str]:
        return [c.name for c in self.columns if not c.is_numeric()]

    def required(self) -> list[str]:
        return [c.name for c in self.columns if c.required]

    def get(self, name: str) -> ColumnSpec:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(f"Unknown column: {name}")


LOAN_SCHEMA = Schema(
    columns=(
        ColumnSpec("Gender", "categorical", allowed_values=("Male", "Female"), nullable=True,
                   description="Applicant gender"),
        ColumnSpec("Married", "categorical", allowed_values=("Yes", "No"), nullable=True,
                   description="Marital status"),
        ColumnSpec("Dependents", "categorical", allowed_values=("0", "1", "2", "3+"), nullable=True,
                   description="Number of dependents"),
        ColumnSpec("Education", "categorical", allowed_values=("Graduate", "Not Graduate"),
                   description="Education level"),
        ColumnSpec("Self_Employed", "categorical", allowed_values=("Yes", "No"), nullable=True,
                   description="Employment status"),
        ColumnSpec("ApplicantIncome", "numeric", minimum=0, maximum=1_000_000,
                   description="Monthly applicant income"),
        ColumnSpec("CoapplicantIncome", "numeric", minimum=0, maximum=1_000_000,
                   description="Monthly co-applicant income"),
        ColumnSpec("LoanAmount", "numeric", minimum=1, maximum=1_000, nullable=True,
                   description="Requested loan amount in thousands"),
        ColumnSpec("Loan_Amount_Term", "numeric", minimum=12, maximum=480, nullable=True,
                   description="Loan term in months"),
        ColumnSpec("Credit_History", "numeric", minimum=0, maximum=1, nullable=True,
                   description="1 = meets credit guidelines, 0 = does not"),
        ColumnSpec("Property_Area", "categorical", allowed_values=("Urban", "Semiurban", "Rural"),
                   description="Property location"),
    )
)

TARGET_COLUMN = "Loan_Status"
FEATURE_COLUMNS = LOAN_SCHEMA.names()
NUMERIC_FEATURES = LOAN_SCHEMA.numeric()
CATEGORICAL_FEATURES = LOAN_SCHEMA.categorical()
