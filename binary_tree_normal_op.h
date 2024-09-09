//
// Created by kai.chen on 2021/12/1.
//
//      1. 遍历、镜像、翻转、
//      2. 二叉搜索树：判断，查 插 删
//
//      3. 统计节点个数
//        3.1 判断完全二叉树
//
//      4. 最近公共祖先
//      5. 通过前序遍历 中序遍历结果 还原二叉树
//
//
//      6. 二叉树中的最大路径和： 见dp_1d  最大子数组和的升级变体

#ifndef DATASTRUCT_ALGORITHM_BINARY_TREE_NORMAL_OP_H
#define DATASTRUCT_ALGORITHM_BINARY_TREE_NORMAL_OP_H
#include <cmath>
#include <vector>
#include <string>
#include <list>
#include <stack>
using namespace std;

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(): val(0),left(nullptr),right(nullptr){}
    explicit TreeNode(int v): val(v), left(nullptr), right(nullptr){}
    TreeNode(int v, TreeNode* l, TreeNode* r): val(v), left(l),right(r){};
};

// 1. 遍历
// topic1: basic traverse
void traverse(TreeNode *root){
    if(root == nullptr){
        return;
    }

    traverse(root->left);
    traverse(root->right);
}

// 相同的二叉树
bool is_same_tree(TreeNode* root1, TreeNode* root2){
    if(root1 == nullptr && root2 == nullptr){
        return true;
    }
    if(root1 == nullptr || root2 == nullptr){
        return false;
    }
    if(root1->val != root2->val){
        return false;
    }
    return is_same_tree(root1->left, root2->left) && is_same_tree(root1->right, root2->right);
}

// 合并二叉树
TreeNode* mergeTrees(TreeNode* t1, TreeNode* t2) {
    if (t1 == nullptr) return t2;
    if (t2 == nullptr) return t1;
    TreeNode* merged = new TreeNode(t1->val + t2->val);
    merged->left = mergeTrees(t1->left, t2->left);
    merged->right = mergeTrees(t1->right, t2->right);
    return merged;
}

// 镜像二叉树
bool is_mirror_tree(TreeNode* root1, TreeNode* root2){
    if(root1 == nullptr && root2 == nullptr){
        return true;
    }
    if(root1 == nullptr || root2 == nullptr){
        return false;
    }
    if(root1->val != root2->val){
        return false;
    }
    return is_mirror_tree(root1->left, root2->right) && is_mirror_tree(root1->right, root2->left);
}

// 对称的二叉树
bool isSymmetric(TreeNode* root) {
    return is_mirror_tree(root, root);
}

// 路径和
bool hasPathSum(TreeNode *root, int sum) {
    if (root == nullptr) {
        return false;
    }
    if (root->left == nullptr && root->right == nullptr) {
        return sum == root->val;
    }
    return hasPathSum(root->left, sum - root->val) ||
           hasPathSum(root->right, sum - root->val);
}

// 翻转二叉树
TreeNode* reverse_tree(TreeNode* root) {
    if (root == nullptr) {
        return nullptr;
    }
    TreeNode* left = reverse_tree(root->left);
    TreeNode* right = reverse_tree(root->right);
    root->left = right;
    root->right = left;
    return root;
}

TreeNode* reverse_tree_vstk(TreeNode* root) {
    if(root == nullptr) return root;
    stack<TreeNode*> stk;
    stk.push(root);
    while(!stk.empty()){
        TreeNode* cur = stk.top(); stk.pop();
        if(cur->left != nullptr) stk.push(cur->left);
        if(cur->right != nullptr) stk.push(cur->right);
        TreeNode* tmp = cur->left;
        cur->left = cur->right;
        cur->right = tmp;
    }
    return root;
}

// 平衡二叉树的判断 见dfs_bfs.h d3

//  2. 二叉搜索树
// topic2: BST Judge, Create Read Update Delete
bool is_valid_bst(TreeNode* root){
    is_valid_bst_helper(root, nullptr, nullptr);
}

bool is_valid_bst_helper(TreeNode* root, TreeNode* min, TreeNode* max){
    if(root == nullptr){
        return true;
    }
    if(min != nullptr && root->val <= min->val){
        return false;
    }
    if(max != nullptr && root->val >= max->val){
        return false;
    }

    return is_valid_bst_helper(root->left, min, root) && is_valid_bst_helper(root->right, root, max);
}

// 查
bool find_in_bst_tree(TreeNode* root, int target){
    if(root == nullptr) {
        return false;
    }
    if(root->val == target) {
        return true;
    }
    else if(root->val < target){
        return find_in_bst_tree(root->right);
    }
    else if(root->val > target){
        return find_in_bst_tree(root->left);
    }
}

// 插
TreeNode* insert_into_bst_tree(TreeNode* root, int value){
    if(root == nullptr){
        return new TreeNode(value);
    }
    if(root->val == value){
        return root;
    }
    else if(root->val < value){
        root->right = insert_into_bst_tree(root->right, value);
    }
    else if(root->val > value){
        root->left = insert_into_bst_tree(root->left, value);
    }
    return root;
}

TreeNode* get_min_node(TreeNode* root){
    if(root == nullptr) return nullptr;
    TreeNode* cur = root;
    while(cur->left != nullptr){
        cur = cur->left;
    }
    return cur;
}

// 删
TreeNode* delete_in_bst_tree(TreeNode* root, int value){
    if(root == nullptr){
        return root;
    }
    if(root->val == value){
        if(root->left == nullptr && root->right == nullptr){
            return nullptr;
        }
        else if(root->left == nullptr){
            return root->right;
        }
        else if(root->right == nullptr){
            return root->left;
        }
        else if(root->left != nullptr && root->right != nullptr){
            TreeNode* min_node_in_right_tree = get_min_node(root->right);
            root->val = min_node_in_right_tree->val;
            root->right = delete_in_bst_tree(root->right, min_node_in_right_tree->val);
        }
    }
    else if(root->val < value){
        root->right = delete_in_bst_tree(root->right, value);
    }
    else if(root->val > value){
        root->left = delete_in_bst_tree(root->left, value);
    }
    return root;
}

// 转累加树
// 思路说明：中序的后面节点即为更大值，累加即可
int sum = 0;
TreeNode* convertBST(TreeNode* root) {
    if(root == nullptr) return root;

    convertBST(root->right); // 反序中序遍历
    sum += root->val;
    root->val = sum;
    convertBST(root->left)

    return root;
}


// 3.统计节点个数
// topic3. count nodes 节点个数
// definition: full binary tree: 二叉树要么有两个子节点要么没有，complete: 完全*， perfect: 满二叉树
int count_nodes_in_tree(TreeNode* root){ // 统计普通二叉树的节点个数，O(N)
    if(root == nullptr) {
        return 0;
    }
    return 1 + count_nodes_in_tree(root->left) + count_nodes_in_tree(root->right);
}

int count_nodes_in_perfect_tree(TreeNode* root){ // 统计满二叉树的节点个数，O(logN)
    int height = 0;
    while(root != nullptr){
        height++;
        root = root->left;
    }
    return (int)pow(2, height) - 1;
}

int count_nodes_in_complete_tree(TreeNode* root){ // 统计完全二叉树的节点个数，O(logN)
    TreeNode* le = root, *ri = root;
    int left_height = 0, right_height = 0;
    while(le != nullptr){
        left_height++;
        le = left->left;
    }
    while(ri != nullptr){
        right_height++;
        ri = ri->right;
    }
    if(left_height == right_height){
        return (int)pow(2, left_height) - 1;
    }
    return 1 + count_nodes_in_complete_tree(root->left) + count_nodes_in_complete_tree(root->right);
}

/*
 * func powerf(x int, n int) int {
	ans := 1
	for n != 0 {
		if n%2 == 1 {
			ans *= x
		}
		x *= x   // 这里是直接双倍的 , 可以打印出x和n的值，自己看看
		n /= 2  // 因为是双倍的，所以只需要进行幂次的一半次数就行了。
	}
	return ans
}
————————————————快速幂---
*/

// 3.1 判断完全二叉树
bool isCompleteTree(TreeNode* root) {
    queue<TreeNode *> q;
    q.push(root);
    while(!q.empty()){
        TreeNode *p = q.front();
        q.pop();
        if(p == nullptr)
            break;
        q.push(p->left);
        q.push(p->right);
    }
    // 前面的 递归遍历 遇到 null就break了
    // 所以到这里只需要，判断 null 后面还有没有节点在 q 里
    while(!q.empty()){
        TreeNode *p = q.front();
        q.pop();
        if(p != nullptr) {
            return false;
        }
    }
    return true;
}



// 4. 最近公共祖先
// 这道题需要先问：是否是二叉搜索树，是否有指向双亲节点的指针？常规的二叉树才用下面的递归方式
// （排序的树可以直接从根节点遍历，都小于往右，都大于往左）
// （有双亲节点的可以转化为两条单链表的第一个公共节点）
 TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
     if(root == nullptr){
         return nullptr;
     }
     if(root == p || root == q){ // 如果发现已经遍历到了p/q，直接返回p/q就是最近祖先，因为这种情况必定是 p/q一个是另一个的祖先
         return root;
     }
     TreeNode* left = lowestCommonAncestor(root->left, p, q);
     TreeNode* right = lowestCommonAncestor(root->right, p, q);

     if(left == nullptr && right == nullptr){ // 左右子树都没找到p/q
         return nullptr;
     }
     if(left != nullptr && right != nullptr){ // 左右子树分别都找到了p/q
         return root;
     }
     return left == nullptr ? right : left;
 }
 // 时间复杂度：O(N): 所有节点都会被访问一次
 // 空间复杂度：O(N)：递归调用的栈深度取决于二叉树的高度,最坏情况下为一条链高度为N
 // dfs是最朴素的做法 -> 这题有更高阶的竞赛做法 ：https://oi-wiki.org/graph/lca/（可以延伸到n个节点的话）一种是倍增算法，一种是借助并查集的Tarjan算法


// 5. 通过前序遍历 中序遍历结果 还原二叉树
// preorder: x| x x x x ~x x
// inorder:  x x x x| x| x x
unordered_map<int, int> Index; // 基于无重复元素
TreeNode* build(vector<int>& pre, vector<int>& in, int pre_left, int pre_right, int in_left, int in_right){
    if(pre_left > pre_right) return nullptr; // 写的时候建议pre和in也定义成员，基于它们没发生改变

    int pre_root = pre_left; // 前序遍历中的第一个节点就是根节点
    int in_root = Index[pre[pre_root]]; // 在中序遍历中定位根节点

    TreeNode* root = new TreeNode(pre[pre_root]); // new一个新根节点
    int size_left_tree = in_root - in_left; // 得到根节点左子树中节点的个数
    // 递归构造左子树
    root->left = build(pre, in, pre_left+1, pre_left+size_left_tree, in_left, in_root-1);
    // 递归构造右子树
    root->right = build(pre, in, pre_left+size_left_tree+1, pre_right, in_root+1, in_right);

    return root;
}
TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
    if(preorder.empty() || inorder.empty() || preorder.size() != inorder.size()){
        return nullptr;
    }
    int n = preorder.size();
    // 构造哈希映射，用来快速定位根节点
    for (int i = 0; i < n; i++) {
        Index[inorder[i]] = i;
    }
    return build(preorder, inorder, 0, n - 1, 0, n - 1);
}
// 时间复杂度：O(n)，其中 n 是树中的节点个数。
// 空间复杂度：O(n)，除去返回的答案需要的 O(n)空间之外，我们还需要使用 O(n) 的空间存储哈希映射，以及 O(h)其中 h 是树的高度)的空间表示递归时栈空间。这里 h < n，所以总空间复杂度为 O(n)。

// 5.延伸 根据 中序遍历 后序遍历结果 还原二叉树
// inorder:  x x x x| x| x x
// postorder: x x x x x ~x | x
unordered_map<int, int> imap;
TreeNode* helper(int in_left, int in_right, int post_left, int post_right, vector<int>& inorder, vector<int>& postorder){
    // 如果这里没有节点构造二叉树了，就结束
    if (in_left > in_right) {
        return nullptr;
    }

    // 选择 post_idx 位置的元素作为当前子树根节点
    int root_val = postorder[post_idx];
    TreeNode* root = new TreeNode(root_val);

    // 根据 root 所在位置分成左右两棵子树
    int index = imap[root_val];

    // 下标减一
    post_idx--;
    // 构造右子树
    root->right = helper(index + 1, in_right, inorder, postorder);
    // 构造左子树
    root->left = helper(in_left, index - 1, inorder, postorder);
    return root;
}
TreeNode* buildTree_zh(vector<int>& inorder, vector<int>& postorder) {
    if(postorder.empty() || inorder.empty() || postorder.size() != inorder.size()){
        return nullptr;
    }
    // 构造哈希映射，用来快速定位根节点
    int n = postorder.size();
    for (int i = 0; i < n; i++) {
        imap[inorder[i]] = i;
    }
    return helper(0, (int)inorder.size() - 1, inorder, postorder);
}


#endif //DATASTRUCT_ALGORITHM_BINARY_TREE_NORMAL_OP_H
