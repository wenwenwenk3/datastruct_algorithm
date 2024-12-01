//
// Created by kai.chen on 2021/12/3.
//
//      1. 反转链表，反转[a,b), K个一组反转链表
//      2. 回文链表
//      3. 两两一组翻转链表
//      4. 链表相交交点
//      5. 删除倒数第n个节点
//       5.1 链表带环，环起点
//       5.2 reverseBetweenMN 递归版本 & 穿针引线
//
//      6. 删除链表的重复元素II（递归+迭代）
//      7. 复制带随机指针的复杂链表
//
//
#ifndef DATASTRUCT_ALGORITHM_LIST_OP_H
#define DATASTRUCT_ALGORITHM_LIST_OP_H
#include <cstdio>
#include <unordered_map>
using namespace std;

// Definition for singly-linked list.
struct ListNode {
     int val;
     ListNode *next;
     ListNode *rand;
     ListNode() : val(0), next(nullptr) {}
     explicit ListNode(int x) : val(x), next(nullptr) {}
     ListNode(int x, ListNode *next) : val(x), next(next) {}
};

ListNode* reverseList(ListNode* head) {
    ListNode* prev = nullptr, *cur = head;
    while(cur!= nullptr){
        ListNode* next = cur->next;
        cur->next = prev;
        prev = cur;
        cur = next;
    }
    return prev;
}

ListNode *reverseBetween(ListNode *head, int left, int right) {
    // 因为头节点有可能发生变化，使用虚拟头节点可以避免复杂的分类讨论
    ListNode *dummyNode = new ListNode(-1);
    dummyNode->next = head;

    ListNode *prev = dummyNode;
    // 第 1 步：从虚拟头节点走 left - 1 步，来到 left 节点的前一个节点
    for (int i = 0; i < left - 1; i++) {
        prev = prev->next;
    }

    // 第 2 步：从 pre 再走 right - left + 1 步，来到 right 节点
    ListNode *rightNode = prev;
    for (int i = 0; i < right - left + 1; i++) {
        rightNode = rightNode->next;
    }

    // 第 3 步：切断出一个子链表（截取链表）
    ListNode *leftNode = prev->next;
    ListNode *curr = rightNode->next;

    // 切断链接
    prev->next = nullptr;
    rightNode->next = nullptr;

    // 第 4 步：反转链表的子区间
    reverseList(leftNode);

    // 第 5 步：接回到原来的链表中
    prev->next = rightNode;
    leftNode->next = curr;
    return dummyNode->next;
}

// reverse List between [a, b)
ListNode* reverseBetweenV2(ListNode* a, ListNode*b){
    ListNode* prev = nullptr, *cur = a;
    while(cur != b){
        ListNode* nxt = cur->next;
        cur->next = prev; // 注意这里和标准reverse的不同，不需要将头的next制成nullptr
        prev = cur;
        cur = nxt;
    }
    return prev;
}

// k个一组翻转链表
ListNode* myreverseKGroup(ListNode* head, int k){
    if(head == nullptr){
        return head;
    }
    ListNode* a = head, *b = head;
    int kcopy = k;
    while(kcopy--){
        if(b == nullptr) return head;
        b = b->next;
    }
    ListNode* newHead = reverseBetween(a, b);
    a->next = myreverseKGroup(b, k);
    return newHead;
}


// 2. 回文链表
ListNode* pleft = nullptr;
bool traverse(ListNode* right){ // 递归版本
    if(right == nullptr) return true;
    bool res = traverse(right->next);
    res = res && (right->val == pleft->val);
    pleft = pleft->next;
    return res;
}
bool isPalindrome(ListNode* head){
    pleft = head;
    return traverse(head);
}

// 1->2->3->2->1
bool isPalindromeV2(ListNode* head) { // 迭代reverse版本
    ListNode* fast = head;
    ListNode* slow = head;
    while(fast != nullptr && fast->next != nullptr){
        // 找中点，偶数偏右
        fast = fast->next->next;
        slow = slow->next;
    }
    // 奇数情况应该在往后走一步
    if(fast != nullptr) slow = slow->next;

    ListNode* left = head;
    ListNode* right = reverseList(slow); // 翻转后半部分
    while(right != nullptr){
        if(left->val != right->val){
            return false;
        }
        left = left->next;
        right = right->next;
    }
    return true;
}

// 3. 两两一组翻转链表
ListNode* swapPairs(ListNode* head) { // 1->2->3->4
    if (!head || !head->next) return head;
    ListNode dummyHead(-1, head);
    ListNode* prev = &dummyHead, *cur = head, *nxt = nullptr;
    while (cur && cur->next) {
        nxt = cur->next; // prev=dummy, cur=1, nxt=2
        prev->next = nxt; // dummy->2
        prev = cur; // prev=cur

        cur->next = nxt->next; // 1->3
        nxt->next = cur; // 2->1
        cur = cur->next;
    }
    return dummyHead.next;
}


// 4. 链表相交
ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
    int lenA = 0, lenB = 0;
    ListNode* curA = headA, *curB = headB;
    while(curA != nullptr) {
        lenA++;
        curA = curA->next;
    }
    while(curB != nullptr) {
        lenB++;
        curB = curB->next;
    }
    curA = headA;
    curB = headB;
    if(lenA > lenB){
        int steps = lenA - lenB;
        while(steps--) curA = curA->next;
        while(curA != nullptr){
            if(curA == curB) return curA;
            curA = curA->next;
            curB = curB->next;
        }
        return nullptr;
    }
    else {
        int steps = lenB - lenA;
        while(steps--) curB = curB->next;
        while(curB != nullptr){
            if(curA == curB) return curA;
            curA = curA->next;
            curB = curB->next;
        }
        return nullptr;
    }
}

//
ListNode* myreverseN(ListNode* head, int n){
    ListNode* prev = head, *cur = head;
    while(n--){
        prev = prev->next;
    }
    ListNode* nN_next = prev;
    while(cur != nN_next){
        ListNode* nxt = cur->next;
        cur->next = prev;
        prev = cur;
        cur = nxt;
    }
    return prev;
}

// 5. 删除倒数第n个节点
ListNode* removeNthFromEnd(ListNode* head, int n) {
    ListNode* dummy = new ListNode(-1, head);
    ListNode* fast = head, *slow = dummy;
    while(n--){
        fast = fast->next;
    }
    while(fast!=nullptr){
        slow = slow->next;
        fast = fast->next;
    }
    // slow 其实是倒数k节点前一个节点
    slow->next = slow->next->next;
    ListNode* res = dummy->next;
    delete dummy;
    return res;
}


// 5.1 链表带环
// 哈希表，快慢指针
bool _hasCycle(ListNode *head) {
    if(head == nullptr) return false;
    ListNode *fast = head, *slow = head;
    while(fast != nullptr && fast->next != nullptr){
        fast = fast->next->next;
        slow = slow->next;
        if(fast == slow) return true;
    }
    return false;
}
// 环起点
ListNode *detectCycle(ListNode *head) {
    ListNode* fast = head, *slow = head;
    bool hasCycle = false;
    while(fast != nullptr && fast->next != nullptr){
        fast = fast->next->next;
        slow = slow->next;
        if(fast == slow) {
            hasCycle = true;
            break;
        }
    }
    if(!hasCycle){ // 无环
        return nullptr;
    }
    slow = head;
    while(slow != fast){
        fast = fast->next;
        slow = slow->next;
    }
    return slow;
}


// reverseBetweenMN 递归版本
// ---- 以下几个递归版本的反转链表仅用于递归练习，效率并不高，时间复杂度是O(N), 但递归栈空间需要O(N)---
ListNode* reverse(ListNode* _head){
    if(_head == nullptr || _head->next == nullptr){
        return _head;
    }
    ListNode* last = reverse(_head->next);
    _head->next->next = _head;
    _head->next = nullptr;
    return last;
}
ListNode* nodeN_next = nullptr;
ListNode* reverseN(ListNode* head, int n){
    if(n == 1){
        nodeN_next = head->next;
        return head;
    }

    ListNode* last = reverseN(head->next, n-1);
    head->next->next = head;
    head->next = nodeN_next;
    return last;
}
ListNode* reverseBetweenMN(ListNode* head, int m, int n){
    if(m == 1){
        return reverseN(head, n);
    }

    head->next = reverseBetweenMN(head->next, m-1, n-1);
    return head;
}

// 穿针引线 reverseBetweenMN (头插法，一次遍历)
ListNode* reverseBetween_ZHEN(ListNode* head, int left, int right){
    ListNode *dummyNode = new ListNode(0);
    dummyNode->next = head;
    ListNode *prev = dummyNode;
    for(int i=0; i<left-1; i++) prev = prev->next;
    ListNode* cur = prev->next;
    for(int i = 0; i < right-left; i++){
        ListNode *nxt = cur->next;
        cur->next = nxt->next;
        nxt->next = prev->next;
        prev->next = nxt;
    }
    return dummyNode->next;
}



//---------------  6. 删除有序链表重复元素II  ---------------------------------------------------
// 题目描述：一个按升序排列的链表，给你这个链表的头节点 head ，请你删除链表中所有存在数字重复情况的节点，只保留原始链表中 没有重复出现 的数字。
// 思路(1) 递归实现，base case是head==null/head->next ==null
//      每次递归只需要判断 head->val是否和head->next->val相等。
//          不相等直接往后走，相等就把head和head->next删掉
ListNode* deleteDuplicates(ListNode* head) {
    if (!head || !head->next) {
        return head;
    }
    if (head->val != head->next->val) {
        head->next = deleteDuplicates(head->next);
    } else {
        ListNode* newHead = head->next;
        while (newHead && head->val == newHead->val) {
            newHead = newHead->next;
        }
        return deleteDuplicates(newHead);
    }
    return head;
} // 时间复杂度O(N), 空间复杂度O(N)

// 思路(2) 迭代实现，
//  一边遍历、一边统计相邻节点的值是否相等，如果值相等就继续后移找到值不等的位置，然后删除值相等的这个区间。
//      删除思路前后指针法，保存pre节点。可以加dummy节点
//  1-> 2-> 2-> 2-> 3
ListNode* deleteDuplicates_v2(ListNode* head) {
    if (!head || !head->next) return head;
    ListNode* preHead = new ListNode(0);
    preHead->next = head;
    ListNode* pre = preHead; // 前后指针
    ListNode* cur = head;
    while (cur) {
        //让cur指向当前重复元素的最后一个位置
        while (cur->next && cur->val == cur->next->val) {
            cur = cur->next;
        }

        if (pre->next == cur) {
            //pre和cur之间没有重复节点，pre后移
            pre = pre->next;
        } else {
            //pre->next指向cur的下一个位置（相当于跳过了当前的重复元素）
            pre->next = cur->next;
        }
        cur = cur->next;
    }
    return preHead->next;
}

// 扩展：忽略链表有序这个性质，使用了两次遍历，第一次遍历统计每个节点的值出现的次数，
// 第二次遍历的时候，如果发现 head.next的 val 出现次数不是 1 次，则需要删除 head.next。
ListNode* deleteDuplicates_vNotOrder(ListNode* head) {
    unordered_map<int, int> m;
    ListNode preHead(0);
    ListNode* prev = &preHead;
    ListNode* cur = head;
    while (cur) {
        m[cur->val]++;
        cur = cur->next;
    }
    cur = head;

    while (cur) {
        if (m[cur->val] == 1) {
            prev->next = cur;
            prev = prev->next;
        }
        cur = cur->next;
    }
    prev->next = nullptr;
    return preHead.next;
}


// 7. 复制带随机指针的复杂链表
// 题目描述：链表除了指向下一节点的指针外，还包含一个随机指针
// 思路：
//    (1). 普通解法可以通过hash表,额外O(N)的空间，时间复杂度O(N)
//         第一步：从左到右遍历，对每一个节点都复制生成对应的副本节点存放在hash表的value中，
//              复制完成后，原链表没有变化，每一个副本的next和rand都指向null
//         第二步：再从左到右遍历，将每一个节点的next和rand指针都设置好
ListNode* copyListWithRand(ListNode* head){
    unordered_map<ListNode*, ListNode*> hmap;
    ListNode* cur = head;
    while(cur != nullptr){
        hmap[cur] = new ListNode(cur->val);
        cur = cur->next;
    }
    cur = head;
    while(cur != nullptr){
        hmap[cur]->next = hmap[cur->next];
        hmap[cur]->rand = hmap[cur->rand];
        cur = cur->next;
    }
    return hmap[head];
}
//    (2). 进阶解法：若限制不能使用额外空间：
//         第一步：从左到右遍历，对每一个节点cur都复制生成对应的copy，然后放在cur与cur->next中间
//         第二步：从左到右遍历，设置每一个副本节点的rand指针。1->1'->2->2'->3->3'->null
//         第三步：只需要将副本链表分离出来就可以了，此时副本的rand已经设置好了
ListNode* copyListWithRand2(ListNode* head){
    if(head == nullptr) return nullptr;
    ListNode* cur = head, *nxt = nullptr;
    // step1. 复制新副本
    while(cur != nullptr){
        nxt = cur->next;  // 1->2->3-null => nxt=2
        cur->next = new ListNode(cur->val); // 复制一个副本插入  1->1'
        cur->next->next = nxt; // 1->1'->2
        cur = nxt; // 2
    }
    cur = head;
    ListNode* curCopy = nullptr;
    // step2. 设置rand指针
    while(cur != nullptr){
        nxt = cur->next->next; //这里每次跳两步
        curCopy = cur->next;
        curCopy->rand = cur->rand != nullptr ? cur->rand->next : nullptr; // 这里cur->rand->next才是cur->rand的副本
        cur = nxt;
    }
    cur = head;
    ListNode* res = head->next;
    // step3. 拆分
    while(cur != nullptr){ // 1->1'->2->2'->3->3'->null
        nxt = cur->next->next; // 2
        curCopy = cur->next;  // 1'
        cur->next = nxt; // 1->2
        curCopy->next = nxt!= nullptr? nxt->next : nullptr; // 这里nxt->next才是nxt的副本
        cur = nxt;
    }
    return res;
}





// --------------- 测试部分 ---------------------------------------------------
class List {
    typedef ListNode Node;
public:
    List():_head(nullptr){}
    ~List(){
        Node* cur = _head;
        Node* next = cur->next;
        while(cur != nullptr){
            next = cur->next;
            delete cur;
            cur = next;
        }
        _head = nullptr;
    }
    void push_back(int value){
        if(_head == nullptr){
            _head = new Node(value);
            return;
        }
        ListNode* last = _head;
        while(last->next != nullptr){
            last = last->next;
        }
        Node* newNode = new Node(value);
        last->next = newNode;
    }
    void print() const{
        ListNode* cur = _head;
        while(cur != nullptr){
            printf("%d", cur->val);
            if(cur->next != nullptr){
                printf("->");
            }
            cur = cur->next;
        }
        printf("\n");
    }

public:
    Node* _head;
};

void test_list_new(){
    List l1;

    for(int i = 1; i < 10; i++){
        l1.push_back(i);
    }
    l1.print();

//
//    for(int i = 9; i > 0; i--){
//        l1.push_back(i);
//    }
//    l1.print();

    ListNode* head = l1._head;
    // ListNode* reverselist = myreverseN(head, 3);

    ListNode* reversKlist = myreverseKGroup(head, 3);
    ListNode* cur = reversKlist;
    while(cur != nullptr){
        printf("(%d)", cur->val);
        if(cur->next != nullptr){
            printf("->");
        }
        cur = cur->next;
    }
    printf("\n");

//
//    ListNode* headN = l1._head;
//    ListNode* reverselistN = reverseN(headN, 5);
//    ListNode* curN = reverselistN;
//    while(curN != nullptr){
//        printf("[%d]", curN->val);
//        if(curN->next != nullptr){
//            printf("->");
//        }
//        curN = curN->next;
//    }
//    printf("\n");

//    ListNode* head = l1._head;
//    bool yes = isPalindromeV2(head);
//    if (yes) {
//        printf("Yes isPalindrome");
//    }
//    else{
//        printf("No isNotPalindrome");
//    }

}


// 排序的循环链表 中插入值，保持非递减排序
ListNode* insert(ListNode* head, int insertVal) {
    ListNode *node = new ListNode(insertVal);
    if (head == nullptr) { // 为空
        node->next = node;
        return node;
    }
    if (head->next == head) { // 只有一个元素
        head->next = node;
        node->next = head;
        return head;
    }
    // 一次遍历
    ListNode *curr = head, *next = head->next;
    while (next != head) {
        if (insertVal >= curr->val && insertVal <= next->val) { // 如果需要插入的值满足可以插在curr,next, 说明找到了目标插入位置
            break;
        }
        if (curr->val > next->val) { // 当curr是升序序部分的末端时
            if (insertVal > curr->val || insertVal < next->val) { // 此时目标插入位置为curr
                break;
            }
        }
        curr = curr->next;
        next = next->next;
    }
    curr->next = node;
    node->next = next;
    return head;
}

#endif //DATASTRUCT_ALGORITHM_LIST_OP_H
