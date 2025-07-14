#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
// avoid name clashes
using std::cout;
using std::endl;
using std::vector;
using std::sort;
using std::numeric_limits;
using std::sqrt;
using std::pow;

// Define a k-dimensional point
// Difference between class and struct:
// Members of a class are private by default, members of a struct are public by default.

struct Point {
    vector<double> coords; // Coordinates of the point
    int id; // Optional identifier for the point

    Point(vector<double> c, int i = -1) : coords(c), id(i) {} // initializer list

    /*An initializer list in C++ is a special syntax used to initialize class member variables before the body of the constructor executes. 
    It allows for efficient initialization.
    Basic use is below:
    public:
    // Constructor with an initializer list
    Example(int x, double y, int z, int& r) 
        : a(x), b(y), c(z), ref(r) { 
        // Constructor body
        std::cout << "Constructor executed\n";
    }
    */

    double distanceSquared(const Point &other) const { // other is a Point and is given by a refernce to avoid unnecessary copying.
                                                       // Also the "const" keyword after the function ensures that the function
                                                       // does not modify the Point object it belongs to.
                                                       // Example usage of the method is below:
                                                    // Point p1({2.0, 3.0});  
                                                    // Point p2({5.0, 4.0});  
                                                    // double distSq = p1.distanceSquared(p2);
        double dist = 0.0;      
        for (size_t i = 0; i < coords.size(); i++) {
            dist += (coords[i] - other.coords[i]) * (coords[i] - other.coords[i]);
        }
        return dist;
    }
};

// Define a k-d tree node
struct KDNode {
    Point point;
    KDNode *left;
    KDNode *right;

    KDNode(Point p) : point(p), left(nullptr), right(nullptr) {} // constructor with an initializer list
};

// KDTree class
class KDTree {
private:
    KDNode *root;
    int k; // Number of dimensions

    // Recursively build the tree
    KDNode* build(vector<Point> &points, int depth = 0) { // Takes a vector filled with Point objects. 
                                                          // depth tracks the current tree level.
                                                          // The function will return a pointer to the root node of the constructed subtree.
        if (points.empty()) return nullptr; // base case (stopping condition)

        int axis = depth % k;               // Level 0 → Split on x (0th dimension).
                                            // Level 1 → Split on y (1st dimension).
                                            // Level 2 → Split on z (2nd dimension) (if k=3).
                                            // Level 3 → Cycle back to x (0th dimension).

        sort(points.begin(), points.end(), [axis](const Point &a, const Point &b) {
            return a.coords[axis] < b.coords[axis];
        }); // sort(points.begin(), points.end(), comparator)
            // How lambda function works: [ capture ] ( parameters ) -> return_type { body }
            // sort modifies the points vector in-place.

        int median = points.size() / 2;
        KDNode* node = new KDNode(points[median]);

        vector<Point> leftPoints(points.begin(), points.begin() + median);
        vector<Point> rightPoints(points.begin() + median + 1, points.end());

        node->left = build(leftPoints, depth + 1);      // node is a pointer to a KDNode object.
                                                        // node->left accesses the left child pointer of node.
                                                        // build() returns a pointer to a KDNode, which gets assigned to node->left.   
        node->right = build(rightPoints, depth + 1);

        return node; // Returns a pointer to the root node of the constructed subtree.
    }

    // Recursively search for the nearest neighbor

    void nearestNeighbor(KDNode* node, const Point &target, KDNode*& best, double &bestDist, int depth = 0) {
        // KDNode* node : Pointer to the current node being examined.
        // const Point &target : The point we want to find the nearest neighbor for.
        // KDNode*& best : A reference to a pointer storing the best (closest) node found so far.
        // double &bestDist : A reference to the best (smallest) squared distance found so far.
        // int depth = 0 : Tracks the current depth in the k-d tree (used to determine splitting axis).



        if (!node) return; // If node == nullptr, it means we've reached a leaf node or an empty subtree.
                           // There’s nothing left to search, so we return immediately.

        double dist = node->point.distanceSquared(target); // node is a pointer to a KDNode object (KDNode*).
                                                           // -> is used to access point (which is a Point object inside KDNode).
                                                           // distanceSquared(target) is a method of Point, so we call it on point.

        if (dist < bestDist) {
            bestDist = dist;
            best = node;
        }

        int axis = depth % k;
        KDNode* nextBranch = (target.coords[axis] < node->point.coords[axis]) ? node->left : node->right;
        KDNode* otherBranch = (nextBranch == node->left) ? node->right : node->left;

        /* axis = depth % k determines the splitting axis (x, y, or z).
           target.coords[axis] is the coordinate of the target point in the current axis.
           node->point.coords[axis] is the coordinate of the current node in the same axis.
           Ternary Operator (? :):
           If the target is smaller than the current node → Search the left subtree first (node->left).
           If the target is larger or equal → Search the right subtree first (node->right).
           nextBranch stores the subtree to explore first. */

        nearestNeighbor(nextBranch, target, best, bestDist, depth + 1);

        /* Checks if we need to explore the other branch.
           Why Do We Check the Other Branch?
           The k-d tree partitions space into two halves at each node.
           The first subtree searched (nextBranch) is the one that contains the target point.
           However, the nearest point may still be in the opposite (otherBranch).
           If the distance to the splitting plane (axisDist) is less than bestDist, we need to check otherBranch. */

        /* What Is the Splitting Plane?

           In a k-d tree, each node divides space into two halves along a splitting axis (x, y, or z).
           The splitting plane is the hyperplane that passes through the node and is perpendicular to the splitting axis. */


        double axisDist = (target.coords[axis] - node->point.coords[axis]) * (target.coords[axis] - node->point.coords[axis]);
        if (axisDist < bestDist) {
            nearestNeighbor(otherBranch, target, best, bestDist, depth + 1);
        }
    }

public:
    KDTree(vector<Point> &points, int dimensions) : k(dimensions) {
        root = build(points);
    }

    // Find the nearest neighbor to a given point
    Point findNearestNeighbor(const Point &target) {
        KDNode* best = nullptr;
        double bestDist = numeric_limits<double>::max();
        nearestNeighbor(root, target, best, bestDist);
        return best ? best->point : Point(vector<double>(k, 0));
        /* Condition: best 
        best is a pointer (KDNode* best).
        If best is not nullptr (i.e., a valid nearest neighbor was found), the condition is true.
        If best == nullptr (no neighbor found), the condition is false. In this case Point(vector<double>(k, 0)) 
        creates a vector<double> of size k where each element is 0.0.
        */
    }
};

// Example usage
int main() {
    vector<Point> points = {
        Point({9.0, 1.0}, 0),
        Point({3.0, 6.0}, 1),
        Point({13.0, 15.0}, 2),
        Point({2.0, 7.0}, 3),
        Point({6.0, 12.0}, 4),
        Point({10.0, 19.0}, 5),
        Point({17.0, 16.0}, 6)
    };

    KDTree tree(points, 2);
    
    Point query({4.0, 8.0});
    Point nearest = tree.findNearestNeighbor(query);

    cout << "Nearest neighbor to (" << query.coords[0] << ", " << query.coords[1] << ") is: "
         << "(" << nearest.coords[0] << ", " << nearest.coords[1] << ") with ID " << nearest.id << endl;

    return 0;
}

// to run:
// on terminal type:     g++ -std=c++11 kdtree.cpp -o kdtree
// and then:             ./kdtree