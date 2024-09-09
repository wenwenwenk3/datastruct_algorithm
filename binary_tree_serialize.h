//
// Created by kai.chen on 2021/12/12.
//
// 1. 序列化&反序列化 serialize & deserialize (即 "遍历" 和 "还原" )
//
// 2. 将二叉搜索树转换为排序双向链表
//  2.1 普通二叉树展开为单链表
//  2.2 将有序链表转换为二叉搜索树
//
#ifndef DATASTRUCT_ALGORITHM_BINARY_TREE_SERIALIZE_H
#define DATASTRUCT_ALGORITHM_BINARY_TREE_SERIALIZE_H
#include <list>
#include <string>
#include <cstdlib>
using namespace std;

struct TreeNode{
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(): val(0), left(nullptr), right(nullptr){};
    explicit TreeNode(int x): val(x), left(nullptr), right(nullptr){};
    TreeNode(int x, TreeNode* l, TreeNode* r): val(x), left(l), right(r){};
};

// 1. 前序遍历方式
string serial_res_of_pre_order;
char NILTag = '#';
char SEPTag = ',';
void serialize_pre_order_helper(TreeNode* root, string& res){ // 前序遍历方式 序列化二叉树
    if(root == nullptr){
        res.push_back(NILTag);
        res.push_back(SEPTag);
        return ;
    }

    res.append(to_string(root->val));
    res.push_back(SEPTag);
    serialize_pre_order_helper(root->left,  res);
    serialize_pre_order_helper(root->right, res);
}
string serialize(TreeNode* root) {
    string ret;
    serialize_pre_order_helper(root, ret);
    return ret;
}

TreeNode* deserialize_helper(list<string>& datalist){
    if(datalist.front() == "#"){
        datalist.erase(datalist.begin());
        return nullptr;
    }

    TreeNode* root = new TreeNode(stoi(datalist.front()));
    datalist.erase(datalist.begin());
    root->left = deserialize_helper(datalist);
    root->right = deserialize_helper(datalist);
    return root;
}
TreeNode* deserialize(string data){ // 反序列化 1,2,4,#,#,#,3,6,#,#,5,#,#,
    list<string> datalist;
    string str;
    for(auto& ch : data){
        if(ch == ','){
            datalist.push_back(str);
            str.clear();
        }
        else{
            str.push_back(ch);
        }
    }
    if(!str.empty()){
        datalist.push_back(str);
        str.clear();
    }
    return deserialize_helper(datalist);
}

// 用字符串数组版本重写（重点掌握版本）
TreeNode* preorder_deserialize_helper(vector<string>& prenums, int l){
    if(prenums[l] == "#" || l > prenums.size()-1) return nullptr;

    TreeNode* root = new TreeNode(stoi(prenums[l]));
    root->left = preorder_deserialize_helper(prenums, 2*l+1);
    root->right = preorder_deserialize_helper(prenums, 2*l+2);
    return root;
}
TreeNode* deserialize_vself(string data) { // 反序列化 1,2,4,#,#,#,3,6,#,#,5,#,#,
    vector<string> nums;
    string num_str;
    for(auto& ch : data){
        if(ch == ',') {
            nums.push_back(num_str);
            num_str.clear();
        }else{
            num_str.push_back(ch);
        }
    }
    if(!num_str.empty()){
        nums.push_back(num_str);
        num_str.clear();
    }
    return preorder_deserialize_helper(nums, 0);
}


// ---------------  测试代码部分   -------------------
TreeNode* preorder_create_tree(vector<int>& prenums, int l){
    if(prenums[l] == -1 || l > prenums.size()-1){
        return nullptr;
    }

    TreeNode* root = new TreeNode(prenums[l]);
    root->left = preorder_create_tree(prenums, 2*l+1);
    root->right = preorder_create_tree(prenums, 2*l+2);

    return root;
}
void pre_order_traverse(TreeNode *root){
    if(root == nullptr){
        return;
    }

    cout<<root->val<<",";
    pre_order_traverse(root->left);
    pre_order_traverse(root->right);
}

void test_serialize() {
    int a[] = {1, 2, 3, 4, -1, 6, 5};

    vector<int> preorder(a, a + sizeof(a) / sizeof(a[0]));
    TreeNode *root = preorder_create_tree(preorder, 0);
    pre_order_traverse(root);
    cout << endl;

    string res_pre_order_serial = serialize(root);
    cout<<res_pre_order_serial<<endl;

    TreeNode* newRoot = deserialize_vself(res_pre_order_serial);
    pre_order_traverse(newRoot);

}


//// (2). 中序遍历方式
//string serial_res_of_in_order;
//string serialize_by_inorder(TreeNode* root){
//    if(root == nullptr){
//        return
//    }
//}
//



// 2. 将二叉搜索树转换为排序双向链表
// 题目描述：将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的节点，只能调整树中节点指针的指向。
// 思路：
//      如果没有限制不能创建节点：中序遍历的方法遍历二叉树将节点保存数组，直接由数组创建双向链表
//      再思考，其实就是 中序遍历中 实现双向链表的push_back

// 借助STL的list就是双向链表很容易实现，但面试肯定让你从0实现一个双向链表
void inorder(TreeNode* root, list<int> &res) {
    if (root == nullptr) {
        return ;
    }
    inorder(root->left, res);
    res.push_back(root->val);
    inorder(root->right, res);
}
list<int> inorderTraversal_dg(TreeNode* root) {
    list<int> res;
    inorder(root, res);
    return res;
}
// 注意上述方法只能拿到排序链表的数据。相当于新建了一条链表，并没有利用原来的节点基础上改指向
// 正确思路：
//      需要修改 *left指向中序前一个节点，*right指向中序后一个节点
//     转换的顺序
void ConvertNode(TreeNode* root, TreeNode** last){
    if(root == nullptr) return;
    TreeNode* cur = root;
    // 处理左子树
    if(cur->left != nullptr) {
        ConvertNode(cur->left, last);
    }

    // 处理当前节点，
    cur->left = *last; // last是已经转换好的链表的最后一个位置
    if(*last != nullptr){
        (*last)->right = cur;
    }
    *last = cur; // 更新last
    // 处理右子树
    if(cur->right != nullptr) {
        ConvertNode(cur->right, last);
    }
}
TreeNode* Convert2Dlink(TreeNode* root){
    TreeNode* last = nullptr; // last是已经转换好的链表的最后一个位置
    ConvertNode(root, &last);
    // 最后需要找出链表的头节点, 因为是双向链表只需要反着遍历一遍
    TreeNode* head = last;
    while(last && last->left != nullptr){// 这里因为root和last都为空
        head = last->left;
    }
    return head;
}


// 2.1 变体：普通二叉树展开为单链表
// 前序遍历，right作为next指针，left统一置空
void pre_traverse(TreeNode* root, vector<TreeNode*> &head) {
    if (root == nullptr) return;
    head.push_back(root);
    pre_traverse(root->left, head);
    pre_traverse(root->right, head);
}
void flatten(TreeNode* root) {
    vector<TreeNode*> head;
    pre_traverse(root, head);
    int n = head.size();
    for (int i = 1; i < n; i++) {
        TreeNode *prev = head[i - 1];
        TreeNode *curr = head[i];
        prev->left = nullptr;
        prev->right = curr;
    }
} // 时间复杂度O(n),空间复杂度O(n)

// 迭代法实现
void flatten_vstk(TreeNode* root) {
    vector<TreeNode*> v;
    stack<TreeNode*> stk;
    TreeNode *node = root;
    while (node != nullptr || !stk.empty()) {
        while (node != nullptr) {
            v.push_back(node);
            stk.push(node);
            node = node->left;
        }
        node = stk.top(); stk.pop();
        node = node->right;
    }
    int size = v.size();
    for (int i = 1; i < size; i++) {
        auto prev = v[i - 1], curr = v[i];
        prev->left = nullptr;
        prev->right = curr;
    }
}
// O(1) 空间的原地 实现，前序会破坏二叉树结构而丢失节点信息。而用后序保存好right就不会丢失right信息
void flatten_inplace(TreeNode* root) {
    if(root == nullptr) return ;
    stack<TreeNode*> stk;
    stk.push(root);
    TreeNode* prev =nullptr;
    while(!stk.empty()){
        TreeNode* cur = stk.top();
        stk.pop();
        if(prev != nullptr) {
            prev->left = nullptr;
            prev->right = cur;
        }
        TreeNode* left = cur->left, *right = cur->right;
        if(right!=nullptr) {
            stk.push(right);
        }
        if(left!=nullptr){
            stk.push(left);
        }
        prev = cur;
    }

}

// 2.2 将有序链表转换为二叉搜索树
// 时间复杂度：O(nlogn)，其中 n 是链表的长度。
// 空间复杂度：O(logn)，递归过程中栈的最大深度。
ListNode* getMedian(ListNode* left, ListNode* right) {
    ListNode* fast = left;
    ListNode* slow = left;
    while (fast != right && fast->next != right) {
        fast = fast->next;
        fast = fast->next;
        slow = slow->next;
    }
    return slow;
}

TreeNode* buildTree(ListNode* left, ListNode* right) {
    if (left == right) {
        return nullptr;
    }
    ListNode* mid = getMedian(left, right);
    TreeNode* root = new TreeNode(mid->val);
    root->left = buildTree(left, mid);
    root->right = buildTree(mid->next, right);
    return root;
}

TreeNode* sortedListToBST(ListNode* head) {
    return buildTree(head, nullptr);
}


// 2.2 将有序数组转换为二叉搜索树
// 思路：每次选中间偏左节点作为根节点
TreeNode* buildTree2(vector<int>& nums, int left, int right) {
    if (left > right) {
        return nullptr;
    }

    // 总是选择中间位置左边的数字作为根节点
    int mid = (left + right) / 2;

    TreeNode* root = new TreeNode(nums[mid]);
    root->left = buildTree2(nums, left, mid - 1);
    root->right = buildTree2(nums, mid + 1, right);
    return root;
}

TreeNode* sortedArrayToBST(vector<int>& nums) {
    return buildTree2(nums, 0, nums.size() - 1);
} // 时间复杂度：O(n), 空间复杂度: O(logn)



#endif //DATASTRUCT_ALGORITHM_BINARY_TREE_SERIALIZE_H
