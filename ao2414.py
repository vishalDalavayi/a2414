import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

data = pd.read_csv('/Users/vishaldalavayi/Downloads/a2414.csv')


data = data.sample(500)  

G = nx.Graph()

threshold = 2  
genre_columns = data.columns[1:]  

for i, row in data.iterrows():
    actor_id = row['actor_id']
    for genre in genre_columns:
        if row[genre] > 0:
            for j, other_row in data.iterrows():
                if i != j and other_row[genre] > 0:
                    if row[genre] + other_row[genre] > threshold:
                        G.add_edge(actor_id, other_row['actor_id'], weight=row[genre] + other_row[genre])


print("Nodes (Actors) in the network:")
for node in G.nodes:
    print(node)

print("\nEdges with weights (Genre connections between actors):")
for edge in G.edges(data=True):
    print(edge)


degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G, k=100) 
closeness_centrality = nx.closeness_centrality(G)


top_degree = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:3]
top_betweenness = sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)[:3]
top_closeness = sorted(closeness_centrality, key=closeness_centrality.get, reverse=True)[:3]


top_nodes = {
    "Degree Centrality": top_degree,
    "Betweenness Centrality": top_betweenness,
    "Closeness Centrality": top_closeness
}

# Convert to DataFrame for display
top_nodes_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in top_nodes.items()]))
print("Top Nodes by Centrality Measures:")
print(top_nodes_df)

# Step 4: Prepare Data for Visualization
top_actors = list(set(top_degree + top_betweenness + top_closeness))

# Create a DataFrame to hold centrality values for top actors
centrality_df = pd.DataFrame({
    "Actor": top_actors,
    "Degree Centrality": [degree_centrality.get(u, 0) for u in top_actors],
    "Betweenness Centrality": [betweenness_centrality.get(u, 0) for u in top_actors],
    "Closeness Centrality": [closeness_centrality.get(u, 0) for u in top_actors]
})

# Step 5: Visualization of Centrality Measures
# Bar plot of centrality measures for top actors
plt.figure(figsize=(10, 6))
sns.barplot(data=centrality_df.melt(id_vars="Actor"), x="Actor", y="value", hue="variable")
plt.title("Centrality Measures for Top Actors")
plt.ylabel("Centrality Value")
plt.xlabel("Actor")
plt.legend(title="Centrality Measure")
plt.show()

# Additional Visualization: Heatmap for Centrality Comparison
plt.figure(figsize=(8, 6))
sns.heatmap(centrality_df.set_index('Actor'), annot=True, cmap="YlGnBu", cbar=True)
plt.title("Centrality Comparison for Top Actors")
plt.xlabel("Centrality Measure")
plt.ylabel("Actor")
plt.show()
