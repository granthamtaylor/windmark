# Windmark

Windmark is a Python module designed for modeling complex, heterogeneous sequential data, where each observation is effectively a data frame. Utilizing a machine learning pipeline, Windmark automates the process of handling sequential data by leveraging the power of Flytekit for workflow orchestration and Polars for efficient data preprocessing. This makes Windmark an ideal choice for data scientists and machine learning engineers who deal with time-series data, nested structures, or any form of sequential data that requires advanced preprocessing and modeling techniques.

## Features

- **Automated Machine Learning Pipeline:** Streamline the process of modeling sequential data with an automated pipeline.
- **Workflow Orchestration with Flytekit:** Utilize Flytekit to manage and scale your machine learning workflows efficiently.
- **Efficient Data Preprocessing with Polars:** Leverage Polars for fast and memory-efficient data preprocessing.
- **Support for Complex, Heterogeneous Sequential Data:** Tailored to handle diverse sequential data formats effectively.

## Installation

To install Windmark, ensure you have Python 3.7 or newer. You can install Windmark directly from PyPI:

```bash
pip install windmark
```

## Quick Start

Below is a simple example to get you started with Windmark. This example demonstrates how to set up a basic machine learning pipeline for sequential data preprocessing and modeling.

```python
import windmark as wm

ledger = "data/mini-ledger.parquet"

split = wm.SequenceSplitter(
    train=0.70,
    validate=0.15,
    test=0.15,
)

schema = wm.Schema.create(

    # required columns
    sequence_id="sequence_id",
    event_id="event_id",
    order_by="event_order",
    target_id="target",

    # unique field names and their datatypes
    use_chip="discrete",
    merchant_state="discrete",
    merchant_city="discrete",
    merchant_name="entity",
    mcc="discrete",
    amount="continuous",
    timedelta="continuous",
    timestamp="temporal",

)

params = wm.Hyperparameters(
    n_steps=50,
    batch_size=64,
    max_epochs=2,
    n_epochs_frozen=1
)

wm.train(
    datapath=ledger,
    schema=schema,
    params=params,
    split=split
)

```

## Backlog

### Features

- [ ] Data quality checks and data exploration visualizations
- [ ] Larger-than-memory data preprocessing (Spark)
- [ ] Support for alternative supervised learning tasks (Regression, Survival)
- [ ] Automatically generated report of model performance
- [ ] Model deployment pipeline

### Experiments

- [ ] Confirm that attention masks are not harming performance
- [ ] Confirm that attention masks are improving compute
- [ ] Explore preprocessing bottlenecks
- [ ] Explore streaming compute bottlenecks

## Documentation

For more detailed documentation on Windmark, including advanced configurations, API details, and custom usage scenarios, please refer to the Windmark Documentation.

## Contributing

We welcome contributions to Windmark! If you're interested in helping improve the project, please check out our Contributing Guidelines for more information on how to get started.

<!-- ## License

Windmark is licensed under the MIT License. -->

## Acknowledgements

Windmark was created to simplify and enhance the process of modeling complex sequential data. We thank the open-source community for the support and contributions that make projects like this possible.

For any questions or support, please open an issue on our GitHub repository, and we'll be happy to help.
