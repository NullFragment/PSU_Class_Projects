from sklearn import tree
import graphviz

features = ['yellow', 'large', 'round']
targets = ['not edible', 'edible']
mushrooms = [[0, 1, 0],
             [1, 1, 1],
             [1, 1, 1],
             [1, 1, 1],
             [1, 1, 1],
             [0, 0, 1],
             [1, 0, 1],
             [1, 1, 0],
             [0, 0, 0],
             [1, 0, 0],
             [1, 1, 1],
             [1, 1, 1],
             [1, 0, 1],
             [1, 0, 1],
             [1, 0, 1],
             [1, 0, 1]]
edible = [[0], [0], [0], [0], [0], [0], [0],
          [1], [1], [1], [1], [1], [1], [1], [1], [1]]

mushroom_clf = tree.DecisionTreeClassifier()
mushroom_clf = mushroom_clf.fit(mushrooms, edible)
dot_data = tree.export_graphviz(mushroom_clf, out_file=None, feature_names=features, class_names=targets, filled=True,
                                rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("mushroom")