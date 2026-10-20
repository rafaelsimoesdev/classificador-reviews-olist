from pydantic import BaseModel, Field


class Comentario(BaseModel):
    texto: str = Field(
        ..., min_length=1, max_length=2000,
        description="Comentário do cliente sobre o pedido",
        examples=["O produto chegou rasgado e até agora ninguém respondeu"],
    )


class Classificacao(BaseModel):
    negativa: bool = Field(description="True se o comentário deve ser tratado como avaliação negativa")
    probabilidade_negativa: float = Field(description="Probabilidade estimada de a avaliação ser negativa (0 a 1)")
    confianca: str = Field(description="ALTA, MÉDIA ou BAIXA, conforme a distância do limiar")
    termos_relevantes: list[str] = Field(description="Termos do comentário que mais pesaram na decisão")
