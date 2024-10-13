from faker import Faker
from windmark.core.orchestration import task


@task
def create_experiment_label() -> str:
    """
    Generates a new label.

    Returns:
        str: The generated label in the format "address:hashtag".
    """

    fake = Faker()

    address = fake.street_name().replace(" ", "-").lower()

    # hashtag = ("").join(random.choice(string.ascii_uppercase) for _ in range(4))

    return address
