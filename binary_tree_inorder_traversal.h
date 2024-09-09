//
// Created by kai.chen on 2021/12/12.
//
// 1. 二叉树的中序遍历
// 这是一道非常经典的常考题：
//  level—1要求：递归、& 非递归(辅助栈)的方式实现
//  level-2要求：非递归实现 + 自己建树测试
//  level-3要求：手撕Morris  （将空间复杂度优化到O(1)）
//   1.1 带标记的创建二叉树
//   1.2 通过有序数组创建BST
//   1.3 二叉搜索树 的第k 大节点
//
// 2. 中序遍历的下一个节点
//    2.1 填充每个节点的下一个右侧节点指针，进阶常量空间
//
// 3. 二叉搜索树迭代器
// 4. 二叉搜索树的后序遍历
// 5. 不同的二叉搜索树种数

// 6. 恢复有两个节点被交换过的二叉搜索树
// 7. 二叉搜索树的最小绝对差

// 监控二叉树 见dp_1d 3.1.1
//
#ifndef DATASTRUCT_ALGORITHM_BINARY_TREE_INORDER_TRAVERSAL_H
#define DATASTRUCT_ALGORITHM_BINARY_TREE_INORDER_TRAVERSAL_H
#include <stack>
#include <vector>
using namespace std;

// Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    explicit TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

// 1. 递归
void inorder(TreeNode* root, vector<int> &res) {
    if (root == nullptr) {
        return ;
    }
    inorder(root->left, res);
    res.push_back(root->val);
    inorder(root->right, res);
}
vector<int> inorderTraversal_dg(TreeNode* root) {
    vector<int> res;
    inorder(root, res);
    return res;
}

// 2. 栈，遍历将root的所有左子树加入栈
vector<int> inorderTraversal_stk(TreeNode* head) {
     vector<int> res;
     TreeNode* root = head;
     stack<TreeNode*> stk;
     // 结束条件是栈为空，且root为空
     while(root != nullptr || !stk.empty()) {
         while(root != nullptr) {
             stk.push(root);
             root = root->left;
         }

         root = stk.top();
         stk.pop();

         res.push_back(root->val);
         root = root->right;
     }
     return res;
}
// 前序遍历
vector<int> preorderTraversal_stk(TreeNode* root) {
    vector<int> res;
    if (root == nullptr) return res;

    stack<TreeNode*> stk;
    TreeNode* node = root;
    while (!stk.empty() || node != nullptr) {
        while (node != nullptr) {
            res.emplace_back(node->val);
            stk.emplace(node);
            node = node->left;
        }
        node = stk.top();
        stk.pop();
        node = node->right;
    }
    return res;
}

TreeNode* preorder_create_tree(vector<int>& prenums, int l){
    if(prenums[l] == -1 || l > prenums.size()-1){
        return nullptr;
    }

    TreeNode* root = new TreeNode(prenums[l]);
    root->left = preorder_create_tree(prenums, 2*l+1);
    root->right = preorder_create_tree(prenums, 2*l+2);

    return root;
}

// 1.2 通过有序数组创建BST，总是选择中间位置右边的数字作为根节点
TreeNode* sortedArrayToBST(vector<int>& nums) {
    return helper(nums, 0, nums.size() - 1);
}

TreeNode* helper(vector<int>& nums, int left, int right) {
    if (left > right) {
        return nullptr;
    }

    // 总是选择中间位置右边的数字作为根节点
    int mid = (left + right + 1) / 2;

    TreeNode* root = new TreeNode(nums[mid]);
    root->left = helper(nums, left, mid - 1);
    root->right = helper(nums, mid + 1, right);
    return root;
}

void pre_order_traverse(TreeNode *root){
    if(root == nullptr){
        return;
    }

    std::cout<<root->val<<",";
    pre_order_traverse(root->left);
    pre_order_traverse(root->right);
}

// 1.3 二叉搜索树 的第k 大节点
// 中序遍历的反向结果
int res, k;
void dfs(TreeNode* root) {
    if(root == nullptr) return;
    dfs(root->right);
    if(k == 0) return;
    if(--k == 0) res = root->val;
    dfs(root->left);
}
int kthLargest(TreeNode* root, int k) {
    this->k = k;
    dfs(root);
    return res;
}


// 3. Morris:
//假设当前遍历到的节点为 curr，将 curr 的左子树中最右边的节点pred的右孩子指向 curr
//
//如果 curr 无左孩子，先将 curr 的值加入答案数组，再访问 curr 的右孩子，即 curr = curr.right。
//如果 curr 有左孩子，则找到 curr 左子树上最右的节点（即左子树中序遍历的最后一个节点，curr 在中序遍历中的前驱节点），我们记为 predecessor。
//根据 predecessor 的右孩子是否为空，进行如下操作。
//  如果 pred 的右孩子为空，则将pred 的右孩子指向 curr，然后访问 curr 的左孩子，即 curr = curr.left。
//  如果 pred 的右孩子不为空，则此时其右孩子指向 curr，说明我们已经遍历完 curr 的左子树，我们将 predecessor 的右孩子置空, \
//      将 curr 的值加入答案数组，然后访问 curr 的右孩子，即 curr = curr.right。
//重复上述操作，直至访问完整棵树。
vector<int> inorderTraversal_morris(TreeNode* root) {
    vector<int> res;
    while (root != nullptr) {
        if (root->left == nullptr){
            res.push_back(root->val);
            root = root->right;
        }
        else if (root->left != nullptr) {
            TreeNode *pred = root->left;
            while (pred->right != nullptr && pred->right != root) {
                pred = pred->right;
            }
            if (pred->right == nullptr) {
                pred->right = root;
                root = root->left;
            } else { // 意味着pred->right == root, 说明遍历完了左子树
                res.push_back(root->val);
                root = root->right;
                pred->right = nullptr;
            }
        }
    }

    return res;
}
void test_inorderTraversal_stk(){
    int a[] = {1, 2, 3, 4, 5, 6};
    vector<int> preorder(a, a + sizeof(a)/sizeof(a[0]));
    TreeNode* root = preorder_create_tree(preorder, 0);
    pre_order_traverse(root);
    cout<<endl;

    vector<int> inorder_tra_result = inorderTraversal_stk(root);
    for(auto it: inorder_tra_result){
        cout<<it<<",";
    }
    cout<<endl;

    vector<int> inorder_tra_result_morris = inorderTraversal_morris(root);
    for(auto it: inorder_tra_result_morris){
        cout<<it<<",";
    }

}

// 2.中序遍历的下一个节点
// 题目描述：节点拥有指向双亲节点的指针
//  思路，根据中序遍历的特点：
//  右子树存在，遍历直到右子树的最左节点
//  右子树不存在，
//          如果是父节点的左节点，父节点就是下一个节点
//          如果不是，往上遍历直到找到cur是父节点的左节点. 一直是父节点的右节点的话，返回null
TreeNode getNext(TreeNode* node){
    if(node == nullptr) return nullptr;
    TreeNode* nxt = nullptr;
    if(node->right != nullptr){ //右子树存在
        nxt = node->right;
        while(nxt->left != nullptr){
            nxt = nxt->left;
        }
    }
    else if(node->parent != nullptr){ //右子树不存在，parent存在
        TreeNode* cur = node;
        TreeNode* par = node->parent;
        while(par != nullptr && cur != par->left){
            // 不是父节点的左节点，往上遍历直到找到是父节点的左节点
            cur = par;
            par = par->parent;
        }
        nxt = par;
    }
    // 右子树不存在，且parent不存在
    return nxt;
}

// 2.1 填充每个节点的下一个右侧节点指针
// 题目描述：填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。
// 思路：BFS层序遍历，建立连接
struct Node {
    int val;
    Node *left;
    Node *right;
    Node *next;
};
Node* connect(Node* root) {
    if (root == nullptr) return root;
    // 初始化队列，将根节点加入队列
    queue<Node*> q;
    q.push(root);

    // 外层的 while 循环迭代的是层数
    while (!q.empty()) {
        int sz = q.size();
        // 遍历这一层的所有节点
        for(int i = 0; i < sz; i++) {
            Node* node = q.front();
            q.pop();
            // 关键位置：建立连接
            if (i < sz - 1) {
                node->next = q.front();
            }
            if (node->left != nullptr) q.push(node->left);
            if (node->right != nullptr) q.push(node->right);
        }
    }
    return root;
} // 时间复杂度: O(N)
//   空间复杂度: O(N)

// 进阶：要求常量级额外空间，可以使用递归栈不算入额外空间
// 思路：遍历过程使用已建立的next指针访问下一个元素，不需要额外的队列存
//    也就是一旦在某层的节点之间建立了 next 指针，那这层节点实际上形成了一个链表。因此，如果先去建立某一层的next 指针，再去遍历这一层，就无需再使用队列了。
// 具体的：
//  (1) 从根节点开始。因为第 0 层只有一个节点，不需要处理。其他层 则在上一层为下一层建立 next 指针。
//      即：位于第 i 层时为第 i+1 层建立 next 指针。一旦完成这些连接操作，继续第 i+1 层为第 i+2 层建立 next 指针
//  (2) 所以每次只要知道下一层的最左边的节点，就可以从该节点开始，像遍历链表一样遍历该层的所有节点

// 把last和p连接上 last->next = p; last = p;
void connectHelper(Node* &last, Node* &p, Node* &nextStart){
    if (last != nullptr) {
        last->next = p;
    }
    last = p;
    if (nextStart == nullptr) { // 只有每一层 开始时 nextStart==null，这也就是保存每层开始最左边的节点
        nextStart = p;
    }
}
Node* connect(Node* root) {
    if (root == nullptr) return root;
    Node *start = root;
    while(start) {
        Node *last = nullptr, *nextStart = nullptr;
        for (Node *p = start; p != nullptr; p = p->next) {
            if (p->left) {
                connectHelper(last, p->left, nextStart); // 把last和p连接上 last->next = p; last = p;
            }
            if (p->right) {
                connectHelper(last, p->right, nextStart);
            }
        }
        start = nextStart;
    }
    return root;
}


// 3. 二叉搜索树迭代器
/**
 * Your BSTIterator object will be instantiated and called as such:
 * BSTIterator* obj = new BSTIterator(root);
 * int param_1 = obj->next();
 * bool param_2 = obj->hasNext();
 */
// 思路(1): 直接在初始化的时候进行中序遍历 保存到一个数组中
//      时间复杂度，初始化操作需要 O(n) 的时间，next操作O(1)，空间复杂度O(n)
// 思路(2): 用栈做迭代
//      时间复杂度：显然，初始化和调用 hasNext() 都只需要 O(1) 的时间。
//          每次调用 next() 函数最坏情况下需要 O(n) 的时间, 但n次调用总共遍历n个节点，平均复杂度为O(1)
//      空间复杂度：最差O(n)，取决于树高度，最差成链表状
class BSTIterator1 {
private:
    void inorder(TreeNode* root, vector<int>& res) {
        if (!root) {
            return;
        }
        inorder(root->left, res);
        res.push_back(root->val);
        inorder(root->right, res);
    }
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        inorder(root, res);
        return res;
    }

    vector<int> arr;
    int idx;
public:
    BSTIterator(TreeNode* root): idx(0), arr(inorderTraversal(root)) {}

    int next() {
        return arr[idx++];
    }

    bool hasNext() {
        return (idx < arr.size());
    }
};

class BSTIterator2 {
    //使用O(h)，保留二叉树的左子树
    stack<TreeNode *> stk;
public:
    // 初始化 BSTIterator 类的一个对象
    BSTIterator(TreeNode* root) {
        dfsLeft(root); // 即只需要dfs保留左子树
    }
    int next() {
        // next = 栈顶 左子树的最后一个节点
        TreeNode * root = stk.front();
        stk.pop();
        int ans = root->val;
        root = root->right;

        dfsLeft(root); // 把栈顶右孩子的 左子树元素 放进栈

        return ans;
    }
    void dfsLeft(TreeNode * root) {
        while (root) {
            stk.push(root);
            root = root->left;
        }
    }
    bool hasNext() {
        return !stk.empty();
    }
};


// 4. 二叉搜索树的后序遍历
// 题目描述：判断一个数组是否可以构成二叉树的后序遍历
// 思路：(1) 递归: 后序遍历 最后一个元素 last 一定是根节点，前面小于last的节点是他的左子树，大于的是右子树
//
bool isValidPostorder(vector<int>& nums, int l, int r){
    if(l >= r) return true;

    // 找到最后一个小于nums[r]的元素
    int m = l;
    while(nums[m] < nums[r]){
        m++;
    } // 退出条件是m为第一个大于nums[r]的元素，即m-1就是最后一个小于的元素

    // 左子树：[l,m-1]  右子树：[m,r-1]  根节点：r
    for(int i = m; i < r; i++){
        if(nums[i] < nums[r]){ // 确认右子树都比nums[r]大
            return false; // 否则 直接return false
        }
    }
    // 到这说明 当前这一层 是可以拆分成 搜索树的后序 结果的。
    // 递归下一层 左右子树
    return isValidPostorder(nums, l, m-1) && isValidPostorder(nums, m, r-1);
}

bool verifyPostorder(vector<int>& postorder){
    return isValidPostorder(postorder, 0, postorder.size() - 1);
}
//时间复杂度 O(N^2)： 每次调用 isValidPostorder(i,j) 减去一个根节点，因此递归占用 O(N) ；最差情况下（即当树退化为链表），每轮递归都需遍历树所有节点，O(N) 。
//空间复杂度 O(N) ： 最差情况下（即当树退化为链表），递归深度将达到 N。


// 5. 不同的二叉搜索树种数
// 题目描述： 给你一个整数 n ，求恰由 n 个节点组成且节点值从 1 到 n 互不相同的 二叉搜索树 有多少种？返回满足题意的二叉搜索树的种数。（1 <= n <= 19）
//      输入：n = 3，输出：5
// 思路： 考虑一个序列1..n，枚举选择数字i为二叉树根节点，1⋯(i−1) 序列作为左子树，(i+1)⋯n 序列作为右子树
//     举例：创建以 3 为根、长度为 7 的不同二叉搜索树，整个序列是 [1, 2, 3, 4, 5, 6, 7]
//      左子序列 [1, 2]构建左子树，从右子序列 [4, 5, 6, 7] 构建右子树，然后将它们组合（即笛卡尔积）
//  设  G(n): 长度为 n 的序列能构成的不同二叉搜索树的个数。
//      F(i,n): 以 i 为根、序列长度为 n 的不同二叉搜索树个数(1≤i≤n)。
//    G(n)= sum(i=1..n)F(i,n)   F(i,n)=G(i−1)⋅G(n−i)
//    ==> G(n)= sum(i=1..n)G(i−1)⋅G(n−i)

int numTrees(int n) {
    vector<int> G(n + 1, 0);
    G[0] = 1;
    G[1] = 1;

    //只需要从小到大计算 G值即可
    for(int j = 2; j <= n; ++j) {
        // i=1..j
        for (int i = 1; i <= j; ++i) {
            G[j] += G[i - 1] * G[j - i];
        }
    }
    return G[n];
}
// 时间复杂度 : O(n^2)，G(n) 函数一共有 n 个值需要求解，每次求解需要 O(n) 的时间复杂度，因此总时间复杂度为 O(n^2)
// 空间复杂度: O(n)。需要 O(n) 的空间存储 G 数组。



// 6. 恢复有两个节点被交换过的二叉搜索树
void inorder(TreeNode* root, vector<int>& nums) {
    if (root == nullptr) {
        return;
    }
    inorder(root->left, nums);
    nums.push_back(root->val);
    inorder(root->right, nums);
}
// 有序序列交换两个元素
// 如果是不相邻的元素的： 那么有两个位置不满足 nums[i] < nums[i+1]
// 如果是相邻的元素： 有一个位置不满足 nums[i] < nums[i+1]
pair<int,int> findTwoSwapped(vector<int>& nums) {
    int n = nums.size();
    int index1 = -1, index2 = -1;
    for (int i = 0; i < n - 1; ++i) {
        if (nums[i + 1] < nums[i]) {
            index2 = i + 1;
            if (index1 == -1) { // 找到一个位置
                index1 = i;
            } else { // 两个位置都找到了
                break;
            }
        }
    }
    int x = nums[index1], y = nums[index2];
    return {x, y};
}

void recover(TreeNode* r, int count, int x, int y) {
    if (r != nullptr) {
        if (r->val == x || r->val == y) {
            r->val = r->val == x ? y : x;
            if (--count == 0) {
                return;
            }
        }
        recover(r->left, count, x, y);
        recover(r->right, count, x, y);
    }
}

void recoverTree(TreeNode* root) {
    vector<int> nums;
    inorder(root, nums);
    pair<int,int> swapped= findTwoSwapped(nums);
    recover(root, 2, swapped.first, swapped.second);
}


// 7. 二叉搜索树的最小绝对差
int res = INT_MAX;
void dfs(TreeNode* root, int& lastnum){
    if(root == nullptr) return;
    dfs(root->left, lastnum);
    if(lastnum == -1){
        lastnum = root->val;
    }else{
        res = min(res, root->val - lastnum);
        lastnum = root->val;
    }
    dfs(root->right, lastnum);
}
int getMinimumDifference(TreeNode* root) {
    int lastnum = -1;
    dfs(root, lastnum);
    return res;
}

#endif //DATASTRUCT_ALGORITHM_BINARY_TREE_INORDER_TRAVERSAL_H
