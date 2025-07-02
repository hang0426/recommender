# test_script.py
from recommender import ProductRecommender

rec = ProductRecommender()
rec.load_data(partner_id=306, category='Apparel & Accessories > Shoes')
rec.preprocess()

recommendations = rec.recommend(
    gender="Men's",
    size=10,
    width="medium",
    brands={'Asics': {'models': ['Gel-Kayano']}},
    colors=['White', 'Blue'],
    top_k=5
)

print(recommendations[['product_id', 'product_name', 'vendor', 'score']])
