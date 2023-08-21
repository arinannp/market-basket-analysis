import os
from kfp import dsl
from kfp.dsl import (Artifact,
                    Dataset,
                    Model,
                    Input,
                    Output,
                    Metrics,
                    component)

project_id = "ps-int-datateamrnd-22072022"
project_region = "asia-southeast2"
bucket_name = "bucket-vertexai-pipeline-artifacts"
template_path = "frequently-bought-together.json"
pipeline_root_path = "gs://bucket-vertexai-pipeline-artifacts/PIPELINE_ROOT_4/"


@component(
    packages_to_install=["pandas", "numpy", "pandas-gbq==0.19.2"],
    base_image="python:3.9"
)
def generate_dataset(
    project_id: str,
    location: str,
    query: str,
    dialect: str,
    dataset: Output[Dataset]
):
    import pandas as pd
    import logging

    logging.getLogger().setLevel(logging.INFO)

    df = pd.read_gbq(
            project_id=project_id, 
            location=location,
            dialect=dialect,
            query=query
        )
    logging.info("Dataframe: ")
    logging.info(df.head())
    logging.info(f"Dataframe shape: {str(df.shape)}")
    logging.info("Dataframe info: ")
    logging.info(df.info())
    logging.info("Dataframe describe: ")
    logging.info(df.describe())
    logging.info("Dataframe null value: ")
    logging.info(df.isnull().sum())
    logging.info("Dataframe number of unique values: ")
    logging.info(df.nunique())
    df.to_csv(dataset.path + ".csv" , index=False, encoding='utf-8-sig')


@component(
    packages_to_install=["pandas", "numpy"],
    base_image="python:3.9"
)
def data_preprocessing(
    dataset: Input[Dataset],
    dataset_clean: Output[Dataset]
):
    import pandas as pd
    import logging

    logging.getLogger().setLevel(logging.INFO)

    df = pd.read_csv(dataset.path + ".csv")

    clean_df = df.dropna(subset = ['ReceiptNo'])
    logging.info("Dataframe remove null value result: ")
    logging.info(clean_df.isnull().sum())

    clean_df = clean_df[(clean_df.Quantity >= 0) & (clean_df.UnitPrice >= 0)]
    logging.info("Dataframe get more than zero quantity: ")
    logging.info(clean_df.describe())

    def has_right_pcode(input):
        import re
        
        x = re.search('^\\d{7}$', input)
        y = re.search('^[a-zA-Z]{1}\\d{7}$', input)
        if (x or y):
            return True
        else:
            return False

    clean_df['ReceiptNo'] = clean_df['ReceiptNo'].astype('str')
    clean_df[clean_df['ReceiptNo'].str.contains("c")].shape[0]
    clean_df['ProductCode'] = clean_df['ProductCode'].astype('str')
    clean_df = clean_df[clean_df['ProductCode'].apply(has_right_pcode) == True]
    logging.info("Dataframe contains right ReceiptNo and ProductCode: ")
    logging.info(clean_df.head())

    df_items = pd.DataFrame(clean_df.groupby('ProductCode').apply(lambda x: x['Product'].unique())).reset_index()
    df_items.rename(columns = { 0: 'ProductClean'}, inplace = True)
    logging.info("ProductCode that have more than one Product: ")
    logging.info(df_items[df_items['ProductClean'].str.len() != 1])

    df_items.loc[:, 'ProductClean'] = df_items.ProductClean.map(lambda x: x[0])
    logging.info(df_items.head())

    clean_df = pd.merge(clean_df, df_items, on = 'ProductCode')
    clean_df = clean_df.drop('Product', axis = 1)
    logging.info("Get only 1 ProductCode equal to 1 Product: ")
    logging.info(clean_df.head())

    clean_df.rename(columns = { 'ProductClean': 'Product'}, inplace = True)
    logging.info("Remaping columns name: ")
    logging.info(clean_df.head())
    df.to_csv(dataset_clean.path + ".csv" , index=False, encoding='utf-8-sig')


@component(
    packages_to_install=["pandas", "numpy", "pandas-gbq==0.19.2", "mlxtend==0.22.0"],
    base_image="python:3.9"
)
def get_recommendation_and_evaluation(
    project_id: str,
    location: str,
    destination_table: str,
    product_category: str,
    dataset: Input[Dataset],
    kpi: Output[Metrics]
):
    import pandas as pd
    import logging
    from datetime import date
    from mlxtend.frequent_patterns import apriori, association_rules

    logging.getLogger().setLevel(logging.INFO)

    df = pd.read_csv(dataset.path + ".csv")
    if product_category == "ALL":
        pass
    else:
        df = df[df['ProductGroup'] == product_category]

    df_items_together = df.groupby(['ReceiptNo','Product'])['Quantity'].sum().sort_values(ascending=False)
    df_items_together = df_items_together.unstack().fillna(0)

    encode = lambda x : True if (x >= 1) else False
    df_items_together = df_items_together.applymap(encode)
    logging.info("Remaping dataframe for modeling: ")
    logging.info(df_items_together.head())

    rec_items = apriori(df_items_together, min_support = 0.01, use_colnames = True, verbose = 1)
    rec_items['length'] = rec_items['itemsets'].apply(lambda x: len(x))
    logging.info("Modeling results: ")
    logging.info(rec_items.sort_values(by=['length'], ascending=False).head(10))
    
    rules_result = association_rules(rec_items, metric = "confidence", min_threshold = 0.0)
    rules_result['date_run'] = str(date.today())
    logging.info("Modeling rule results: ")
    logging.info(rules_result.sort_values(by=['confidence'], ascending=False))
    
    kpi.log_metric("avg antecedent support", float(rules_result.loc[:, 'antecedent support'].mean()) if float(rules_result.loc[:, 'antecedent support'].mean()) != float('inf') else 0)
    kpi.log_metric("avg consequent support", float(rules_result.loc[:, 'consequent support'].mean()) if float(rules_result.loc[:, 'consequent support'].mean()) != float('inf') else 0)
    kpi.log_metric("avg support", float(rules_result.loc[:, 'support'].mean()) if float(rules_result.loc[:, 'support'].mean()) != float('inf') else 0)
    kpi.log_metric("avg confidence", float(rules_result.loc[:, 'confidence'].mean()) if float(rules_result.loc[:, 'confidence'].mean()) != float('inf') else 0)
    kpi.log_metric("avg lift", float(rules_result.loc[:, 'lift'].mean()) if float(rules_result.loc[:, 'lift'].mean()) != float('inf') else 0)
    kpi.log_metric("avg leverage", float(rules_result.loc[:, 'leverage'].mean()) if float(rules_result.loc[:, 'leverage'].mean()) != float('inf') else 0)
    kpi.log_metric("avg conviction", float(rules_result.loc[:, 'conviction'].mean()) if float(rules_result.loc[:, 'conviction'].mean()) != float('inf') else 0)
    kpi.log_metric("avg zhangs metric", float(rules_result.loc[:, 'zhangs_metric'].mean()) if float(rules_result.loc[:, 'zhangs_metric'].mean()) != float('inf') else 0)
    
    rules_result["antecedents"] = rules_result["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
    rules_result["consequents"] = rules_result["consequents"].apply(lambda x: list(x)[0]).astype("unicode")
    rules_result["antecedent_support"] = rules_result["antecedent support"]
    rules_result["consequent_support"] = rules_result["consequent support"]
    rules_result.drop(['antecedent support', 'consequent support'], axis=1, inplace=True)
    rules_result = rules_result.astype(str)
    rules_result = rules_result[['date_run','antecedents','consequents','antecedent_support','consequent_support','support','confidence','lift','leverage','conviction','zhangs_metric']]
    rules_result.to_gbq(project_id=project_id, location=location, destination_table=destination_table, if_exists="replace")




@dsl.pipeline(
    name="koufu market basket analysis",
    description="Frequently Bought Together Model",
    pipeline_root=pipeline_root_path)
def pipeline(
        bucket_name: str="your-bucket-name", 
        project_id: str ="your-project-id",
        location: str ="your-project-region",
        args: list = ["--arg-key", "arg-value"],
    ):

    generate_dataset_op = generate_dataset(
        project_id=project_id, 
        location=location, 
        dialect="standard", 
        query="""
            SELECT
                Receipt_No AS ReceiptNo,
                Receipt_Created AS ReceiptDate,
                Product_Code AS ProductCode,
                Product_Group AS ProductGroup,
                Product,
                Qty AS Quantity,
                Unit_Price AS UnitPrice,
                Qty * Unit_Price AS TotalPrice
            FROM
                `ps-int-datateamrnd-22072022.koufu_product_trx_list.product_list`
        """)
    
    data_preprocessing_op = data_preprocessing(dataset=generate_dataset_op.outputs["dataset"])

    recommendation_matrix_eval_all_op = get_recommendation_and_evaluation(
        project_id=project_id, 
        location=location,
        destination_table="koufu_product_trx_list.cust_recomm_all_category",
        product_category="ALL",
        dataset=data_preprocessing_op.outputs["dataset_clean"]
    )
    recommendation_matrix_eval_hot_drink_op = get_recommendation_and_evaluation(
        project_id=project_id, 
        location=location,
        destination_table="koufu_product_trx_list.cust_recomm_hot_drink_category",
        product_category="HOT DRINK",
        dataset=data_preprocessing_op.outputs["dataset_clean"]
    )
    recommendation_matrix_eval_drink_consumables_op = get_recommendation_and_evaluation(
        project_id=project_id, 
        location=location,
        destination_table="koufu_product_trx_list.cust_recomm_drink_consumables_category",
        product_category="DRINK CONSUMABLES",
        dataset=data_preprocessing_op.outputs["dataset_clean"]
    )
    recommendation_matrix_eval_shelf_op = get_recommendation_and_evaluation(
        project_id=project_id, 
        location=location,
        destination_table="koufu_product_trx_list.cust_recomm_shelf_category",
        product_category="SHELF",
        dataset=data_preprocessing_op.outputs["dataset_clean"]
    )
    recommendation_matrix_eval_beer_op = get_recommendation_and_evaluation(
        project_id=project_id, 
        location=location,
        destination_table="koufu_product_trx_list.cust_recomm_beer_category",
        product_category="BEER",
        dataset=data_preprocessing_op.outputs["dataset_clean"]
    )
    # recommendation_matrix_eval_premium_food_op = get_recommendation_and_evaluation(
    #     project_id=project_id, 
    #     location=location,
    #     destination_table="koufu_product_trx_list.cust_recomm_premium_food_category",
    #     product_category="PREMIUM FOOD",
    #     dataset=data_preprocessing_op.outputs["dataset_clean"]
    # )
    # recommendation_matrix_eval_cold_drink_op = get_recommendation_and_evaluation(
    #     project_id=project_id, 
    #     location=location,
    #     destination_table="koufu_product_trx_list.cust_recomm_cold_drink_category",
    #     product_category="COLD DRINK",
    #     dataset=data_preprocessing_op.outputs["dataset_clean"]
    # )
    # recommendation_matrix_eval_cigarette_op = get_recommendation_and_evaluation(
    #     project_id=project_id, 
    #     location=location,
    #     destination_table="koufu_product_trx_list.cust_recomm_cigarette_category",
    #     product_category="CIGARETTE",
    #     dataset=data_preprocessing_op.outputs["dataset_clean"]
    # )
    # recommendation_matrix_eval_misc_op = get_recommendation_and_evaluation(
    #     project_id=project_id, 
    #     location=location,
    #     destination_table="koufu_product_trx_list.cust_recomm_misc_category",
    #     product_category="MISC",
    #     dataset=data_preprocessing_op.outputs["dataset_clean"]
    # )
    # recommendation_matrix_eval_premium_drink_op = get_recommendation_and_evaluation(
    #     project_id=project_id, 
    #     location=location,
    #     destination_table="koufu_product_trx_list.cust_recomm_premium_drink_category",
    #     product_category="PREMIUM DRINK",
    #     dataset=data_preprocessing_op.outputs["dataset_clean"]
    # )
    # recommendation_matrix_eval_dim_sum_op = get_recommendation_and_evaluation(
    #     project_id=project_id, 
    #     location=location,
    #     destination_table="koufu_product_trx_list.cust_recomm_dim_sum_category",
    #     product_category="DIM SUM",
    #     dataset=data_preprocessing_op.outputs["dataset_clean"]
    # )





def compile_pipeline():
    from kfp.compiler import Compiler

    Compiler().compile(
        pipeline_func=pipeline,
        package_path=template_path
    )


def run_pipeline():
    import google.cloud.aiplatform as aip

    aip.init(
        project=project_id,
        location=project_region,
    )

    job = aip.PipelineJob(
        display_name="koufu market basket analysis",
        template_path=template_path,
        pipeline_root=pipeline_root_path,
        enable_caching=False,
        parameter_values={
            "bucket_name": bucket_name, 
            "project_id": project_id,
            "location": project_region,
            "args": []
        }
    )

    job.submit(service_account="vertexai-pipeline-sa@ps-int-datateamrnd-22072022.iam.gserviceaccount.com")


if __name__ == '__main__':
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Pointstar/Koufu Beverage/Frequently Bought Together ML/vertexai-pipeline-sa.json"

    compile_pipeline()
    run_pipeline()