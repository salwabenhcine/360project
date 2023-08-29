from django.urls import path , include
from app import views

urlpatterns = [
    path('marque/', views.Marque),
    path('produit/', views.Produits),

]