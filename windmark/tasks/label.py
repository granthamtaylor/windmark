# Copyright Grantham Taylor.

from faker import Faker
from windmark.orchestration.environments import context


@context.default(cache=False)
def create_experiment_label() -> str:
    """
    Generates a new label.

    Returns:
        str: The generated label in the format "address".
    """

    fake = Faker()

    return fake.street_name().replace(" ", "-").lower()
