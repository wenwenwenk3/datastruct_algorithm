//
// Created by kai.chen on 2021/11/25.
// 排序算法
//      1. 插入排序
//      2. 选择排序
//      3. 冒泡排序
//      4. 希尔插入排序
//      5. 堆排序
//      6. 归并排序
//          6.1 合并有序数组到原较长数组a[]
//          6.2 计算数组的小和
//          . 数组中的逆序对 见array_string.h 7.
//      7. 快排
//              // 其他线性排序（桶排，计数排，基数排）：https://blog.csdn.net/qq_25026989/article/details/89367954
//              // 其他不常见排序：Cocktail sort鸡尾酒排序、Gnome Sort地精排序、Bitonic sort(双调排序)、Bogo sort(猴子排序)、Wiggle(摆动排序)
//      8. 链表排序:
//          8.1 链表插入排序
//          8.2 链表归并排序 (自顶向下，自底向上)
//             - 合并排序好的链表 合并升序数组在 find_kth_largest 3
//          8.3 链表奇升偶降排序
//          8.4 重排链表
//          8.5 奇偶链表
//          8.6 循环有序链表中插入值
//      9. 调整数组使奇数位于偶数前面，双指针
//
#ifndef DATASTRUCT_ALGORITHM_SORT_H
#define DATASTRUCT_ALGORITHM_SORT_H
#include <vector>
#include "list_op.h"
#include <queue>

using namespace std;

inline void swap(int *a, int *b){
    int temp = *a;
    *a = *b;
    *b = temp;
}

// 1.插入排序。（假设前n个有序，依次往前碰找到该插入的位置插入）
void InsertSort(int a[], int n){
    for(int i = 1; i < n; i++){
        int j;
        int cur = a[i];
        for(j = i-1; j >= 0 && a[j] > cur; j--){
            a[j+1] = a[j];
        }
        a[j+1] = cur;
    }
}

// 2 4 5, 3 4

// 2.选择排序。（选择出后续最小值与当前值交换）
void SelectSort(int a[], int n){
    for(int i = 0; i < n-1; i++){
        int minIdx = i;
        for(int j = i+1; j < n; j++){
            if(a[j] < a[minIdx]){
                minIdx = j;
            }
        }
        swap(&a[i], &a[minIdx]);
    }
}
// 选择排序不稳定的原因
//比如A 80 B 80 C 70 这三个卷子从小到大排序
//第一步会把C和A做交换 变成C B A
//第二步和第三步不需要再做交换了。所以排序完是C B A
//但是稳定的排序应该是C A B

// 3.冒泡排序
void BubbleSort(int a[], int n){
    for(int i = n-1; i > 0 ; i--){
        for(int j = 0; j < i; j++){
            if(a[j+1] < a[j]){
                swap(a[j+1], a[j]);
            }
        }
    }
}
// 稳定的排序算法，冒泡可以控制小于的时候才交换，且不会轻易改变后续值

// 4. 希尔排序 [高级插入排序，分组插入]
void shell_sort_helper(int a[], int n, int gap) {
    for (int i = gap; i < n; i++) {
        int j = i - gap, temp = a[i];
        for (; j >= 0 && a[j] > temp; j= j-gap) {
            a[j+gap] = a[j];
        }
        a[j+gap] = temp;
    }
}

void shell_sort(int a[], int n){
    for(int gap = n/2; gap > 0; gap /= 2){
        shell_sort_helper(a, n, gap);
    }
}

// 5. 堆排序 [高级选择排序，通过堆进行选择]
void adjust_down(int a[], int n, int root){ // 往下调一个元素
    int parent = root;
    int child = parent*2 + 1; // left child
    while(child < n){
        if(child+1 < n && a[child+1] > a[child]){ // select min of (left child, right child)
            child = child +1;
        }
        if(a[child] > a[parent]){ // if child > parent, swap child/parent
            swap(a[child], a[parent]);
            parent = child;
            child = parent*2 + 1;
        }
        else{
            break;
        }
    }
}

void heap_sort(int a[], int n){
    // 第一步：建堆
    for(int i = (n-1-1)/2; i >= 0; i--){
        adjust_down(a, n, i);
    }
    // 第二步：排升序，建大堆依次调整
    for(int i = n-1; i > 0; i--){
        swap(&a[i], &a[0]);
        adjust_down(a, i, 0);
    }
}

// 6. 归并排序
// temp[]为存放两组比较结果的临时数组，最后会排好序的数填回到array数组的对应位置
void mergeArray(int array[], int first, int mid, int last, int temp[]) {
    int i = first, j = mid + 1; // i为第一组的起点, j为第二组的起点
    int m = mid, n = last; // m为第一组的终点, n为第二组的终点
    int k = 0; // k用于指向temp数组当前放到哪个位置
    while (i <= m && j <= n) { // 将两个有序序列循环比较, 填入数组temp
        if (array[i] <= array[j])
            temp[k++] = array[i++];
        else
            temp[k++] = array[j++];
    }
    while (i <= m) { // 如果比较完毕, 第一组还有数剩下, 则全部填入temp
        temp[k++] = array[i++];
    }
    while (j <= n) {// 如果比较完毕, 第二组还有数剩下, 则全部填入temp
        temp[k++] = array[j++];
    }
    for (i = 0; i < k; i++) {// 将排好序的数填回到array数组的对应位置
        array[first + i] = temp[i];
    }
}
// 归并
void MergeSort(int array[], int first, int last, int temp[]) {
    if (first < last) {
        int mid = first + (last -first) / 2;
        // 递归归并左边元素
        MergeSort(array, first, mid, temp);
        // 递归归并右边元素
        MergeSort(array, mid + 1, last, temp);
        // 再将二个有序数组合并
        mergeArray(array, first, mid, last, temp);
    }
}
void mergeSort(int a[], int n){
    int* temp = (int *)malloc(n * sizeof(int));
    MergeSort(a, 0, n-1, temp);
}

// 归并变体1：合并有序数组到原较长数组a[]
void merge_b1(int a[], int m, int b[], int n){
    // 合并到a数组中，a数组足够长。思路从后往前填值
    int i = m-1, j = n-1, p = m+n-1;
    while(~i && ~j) { // ～取反 等价于i>=0; 因为-1取反全0
        a[p--] = a[i]>b[j] ? a[i--]:b[j--];
    }
    // b数组还没有跑到-1，直接填充到前面
    while(~j){
        a[p--] = b[j--];
    }
}
void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
    // 合并到a数组中，a数组足够长。思路从后往前填值
    int i = m-1, j = n-1, p = m+n-1;
    while(~i && ~j) { // ～取反 等价于i>=0; 因为-1取反全0
        nums1[p--] = nums1[i]>nums2[j] ? nums1[i--]:nums2[j--];
    }
    // b数组还没有跑到-1，直接填充到前面
    while(~j){
        nums1[p--] = nums2[j--];
    }
}

// 归并变体2：合并区间
// 题目描述：输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
//      输出：[[1,6],[8,10],[15,18]]
// 思路：排序，然后将第一个区间添加如res，接下来迭代考虑下面的区间
//     如果新区间的左端点大于当前区间的右端点，那么不会有重合，直接push_back
//     否则，有重合部分，只需要将当前区间右端点更新为两个区间的右端点较大值
vector<vector<int>> merge_b2(vector<vector<int>>& intervals) {
    if(intervals.size() == 0) return {};
    sort(intervals.begin(), intervals.end());
    vector<vector<int>> res;
    res.push_back({intervals[0][0], intervals[0][1]});
    for(int i = 1; i < intervals.size(); i++){
        int left = intervals[i][0], right = intervals[i][1];
        if(left > res.back()[1]){ // left > 当前区间的右端点，不会有重复区间
            res.push_back({left, right});
        }
        else{
            res.back()[1] = max(res.back()[1], right);
        }
    }
    return res;
}

// 归并变体3: 计算数组的小和
// 题目描述：在一个数组中，每一个数左边比当前数小的数累加起来，叫做这个数组的小和。
// 思路： 暴力解法是O(n^2), 可以借助归并的思路
// smallSum([1,3,4,2,5])实际就等于smallSum([1,3,4])+smallSum([2,5])+c: 之所以还有个c，是因为左半段数组中可能存在比右半段数组小的元素，这个不能遗漏。
// 在merge时，左段数组L和右段数组R都是有序的了。
// 结合下边的示意图，当L[i]<=R[j]时，表示L[i]比R[j]~R[r]的元素都要小。因此c需加上j及以后元素的个数*L[i]，即c+=(r-j+1) * L[i]。
typedef long long LL;
const int N = 1e5 + 10;
int nums[N];
int temp[N];
//long long是因为结果可能爆int
LL merge(int a[], int l, int mid, int r){
    int i = l, j = mid + 1, k = 0;
    LL res = 0;
    while (i <= mid && j <= r){
        if (a[i] <= a[j]){
            res += (r - j + 1) * a[i];
            temp[k++] = a[i++];
        }
        else temp[k++] = a[j++];
    }
    while (i <= mid) temp[k++] = a[i++];
    while (j <= r) temp[k++] = a[j++];

    for (i = l, k = 0; i <= r; i++){
        a[i] = temp[k++];
    }
    return res;
}

LL getSmallSum(int a[], int l, int r){
    if (l == r) return 0;
    int mid = (l + r) / 2;
    LL L = getSmallSum(a, l, mid), R = getSmallSum(a, mid + 1, r);
    LL c = merge(a, l, mid, r);
    return L + R + c;
} // 时间复杂度：O(N * logN), 空间复杂度：O(N)

void test_getSmallSum(){
    int n;
    cin >> n;
    for (int i = 0; i < n; i++) cin >> nums[i];
    cout << getSmallSum(nums, 0, n - 1) << endl;
}

// 7. 快速排序
// 快速排序是先将一个元素排好序，然后再将剩下的元素排好序。
// 快速排序的核心是 partition 函数，
//   partition 函数的作用是在 nums[l..r] 中寻找一个切分点 p，通过交换元素使得 nums[lo..p-1] 都小于等于 nums[p]，且 nums[p+1..hi] 都大于 nums[p]：
int partitionl(int a[], int n, int left, int right){
    int pivot = a[right];
    int l = left, r = right;
    while(l < r){
        while(l < r && a[l] <= pivot) l++;
        while(l < r && pivot <= a[r]) r--;
        swap(&a[l], &a[r]);
    }
    swap(&a[l], &a[right]);
    return l;
}

void quick_sort(int a[], int n, int start, int end){
    if(start >= end) return;
    int div = partitionl(a, n, start, end);
    quick_sort(a, n, start, div-1);
    quick_sort(a, n, div+1, end);
}// 时间复杂度：最差 N^2
// 当数据基本趋于有序的时候，每次基准值都是最后一位，区间没有被划分开。所以算法最差的时间复杂度为O(N^2).

vector<int> sortArray(vector<int>& nums) {
    int *a = nums.data();
    int n = nums.size();
    quick_sort(a, n, 0, n-1);

    return vector<int>(a, a+n);
}

// 8. 链表插入排序
//  假设前面部分有序，每一个找插入位置。挪指针
//   dummy -> x1 -> x3 -> |x2 -> x4
ListNode* insertionSortList(ListNode* head) {
    if (head == nullptr) {
        return head;
    }
    ListNode* dummy = new ListNode(0);
    dummy->next = head; // 添加dummy节点方便在头节点前插入节点
    ListNode* sortedIndex = head;
    ListNode* curr = head->next;
    while (curr != nullptr) {
        // 当前节点比sorted的最大元素还大，直接往后走
        if (sortedIndex->val <= curr->val) {
            sortedIndex = sortedIndex->next;
        }else { // 否则寻找插入位置
            ListNode *prev = dummy;
            // 直到找到要插入的位置（第一个大于curr的节点前面）x3
            while (prev->next->val <= curr->val) {
                prev = prev->next;
            }

            sortedIndex->next = curr->next; // x3->x4
            curr->next = prev->next; // x2 -> x3
            prev->next = curr; // x1 -> x2
        }
        curr = sortedIndex->next; // cur = x4
    }
    return dummy->next;
}

// 8.2 链表归并排序
//  思路：找到链表的中点，以中点为分界，将链表拆分成两个子链表。
//      对两个子链表分别排序。
//      将两个排序后的子链表合并，参考合并升序链表
// 递归版merge sortedlist实现，易理解,  迭代版实现见下面补充
ListNode* mergelist(ListNode* p1, ListNode* p2){
    if(p1 == nullptr) return p2;
    if(p2 == nullptr) return p1;
    ListNode* ph;
    if(p1->val < p2->val){
        ph = p1;
        ph->next = mergelist(p1->next, p2);
    }else{
        ph = p2;
        ph->next = mergelist(p1, p2->next);
    }
    return ph;
}
// 排序链表区间 [head, tail)
ListNode* sortList(ListNode* head, ListNode* tail) {
    if (head == nullptr) {
        return head;
    } // base case
    if (head->next == tail) {
        head->next = nullptr;
        return head;
    }
    // 快慢指针找中点
    ListNode* slow = head, *fast = head;
    while (fast != tail) { //快指针每次移动 2 步，慢指针每次移动 1 步，当快指针到达链表末尾时，慢指针指向的链表节点即为链表的中点。
        slow = slow->next;
        fast = fast->next;
        if (fast != tail) {
            fast = fast->next;
        }
    }
    ListNode* mid = slow;
    // 拆分左右区间
    return mergelist(sortList(head, mid), sortList(mid, tail));
} // 时间复杂度O(N*logN), 空间复杂度是O(logN * logN),递归栈需要的深度

ListNode* sortList(ListNode* head) {
    return sortList(head, nullptr);
}

// 补充，合并排序好的链表可以用迭代的方式，不需要第二层递归栈空间，总空间复杂度O(logN)
ListNode* merge(ListNode* head1, ListNode* head2) {
    ListNode* dummyHead = new ListNode(0);
    ListNode* temp = dummyHead, *temp1 = head1, *temp2 = head2;
    while (temp1 != nullptr && temp2 != nullptr) {
        if (temp1->val <= temp2->val) {
            temp->next = temp1;
            temp1 = temp1->next;
        } else {
            temp->next = temp2;
            temp2 = temp2->next;
        }
        temp = temp->next;
    }
    if (temp1 != nullptr) {
        temp->next = temp1;
    } else if (temp2 != nullptr) {
        temp->next = temp2;
    }
    return dummyHead->next;
}

// 进阶： 合并 k 个升序链表
// 题目描述：给你一个升序链表数组，求合并后的升序链表
//  输入：[1,4,5],[1,3,4],[2,6]
//  输出：[1,1,2,3,4,4,5,6]
// 思路：
//  直观做法：比较每个链表的头节点，每次选取最小的放入结果
//    这相当于每次需要求 k 个链表的头节点最小值，可以想到堆
struct cmp{
    bool operator()(ListNode* a, ListNode* b){
        return a->val > b->val; // pq默认头部是最大元素
    }
};
ListNode* mergeKLists(vector<ListNode*>& lists) {
    ListNode* dummy = new ListNode(-1);
    ListNode* tmp = dummy;
    // 利用priority_queue可以内部实现堆调整
    // priority_queue<Type, Container, Functional>
    priority_queue<ListNode*, vector<ListNode*>, cmp> pq;
    for(auto& node : lists){
        if(node != nullptr) pq.push(node);
    }
    while(!pq.empty()){
        ListNode* curhead = pq.top();
        pq.pop();
        tmp->next = curhead;
        tmp = tmp->next;
        if(curhead->next != nullptr){
            pq.push(curhead->next);
        }
    }
    return dummy->next;
}



// 8.3 链表奇升偶降排序
// 题目描述：给定一个奇数位升序，偶数位降序的链表，将其重新排序。
//      输入: 1->8->3->6->5->4->7->2->NULL
//      输出: 1->2->3->4->5->6->7->8->NULL
// 思路：
//    思路(1)
//      按奇偶位置拆分链表 (得1->3->5->7->NULL和8->6->4->2->NULL
//     + 翻转偶链表 (得1->3->5->7->NULL和2->4->6->8->NULL
//     + 合并升序链表 (得1->2->3->4->5->6->7->8->NULL
void reorderListEvenOdd1(ListNode *head) {
    if(head == nullptr || head->next == nullptr){
        return ;
    }
    //拆分出奇节点链表和偶节点链表
    ListNode* odd = head;
    ListNode* evenHead = head->next;
    ListNode* even = evenHead;
    while(even != nullptr && even->next != nullptr){
        odd->next = even->next;
        odd = odd->next;
        even->next = odd->next;
        even = even->next;
    }
    odd->next = nullptr; //当链表长度为偶数时，odd.next还指向最后一个偶节点，需要断开

    //反转偶节点链表
    evenHead = reverse(evenHead);
    // 合并升序链表
    head = mergelist(head, evenHead);
}

// 思路(2) 用数组保存的方法：
// 只需要把偶数位(奇数下标)的链表翻转
void reorderListEvenOdd2(ListNode *head) {
    if (head == nullptr) return ;

    vector<ListNode *> nodesArray;
    ListNode *cur = head;
    while (cur != nullptr) {
        nodesArray.push_back(cur);
        cur = cur->next;
    }

    int sz = nodesArray.size();
    int left = 0, right = sz - 1;

    while(left < right){
        if(left % 2 != 0){ // 奇数下标位置1 3 5， 需要交换
            ListNode* temp = nodesArray[left];
            nodesArray[left] = nodesArray[right];
            nodesArray[right] = temp;
            left++;
            right-=2; // next 偶数位置(从后往前数)
        }
        else left++; // 偶数下标位置，不用交换
    }

    for(int i = 0; i < sz-1; i++){ // 这里有点问题，需要保证1，2，3，这样压着走才行
        nodesArray[i]->next = nodesArray[i+1];
    }
    nodesArray[sz-1]->next = nullptr;
}


// 8.4 重排链表
//      输入：L0 → L1 → … → Ln - 1 → Ln
//      输出：L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
//  方法(1). 新建一个数组，在数组中重排, 牺牲O(N)空间
//  方法(2). 寻找链表中点 + 链表逆序 + 合并链表
void reorderList11(ListNode *head) {
    if (head == nullptr) return ;

    vector<ListNode *> nodesArray;
    ListNode *cur = head;
    while (cur != nullptr) {
        nodesArray.push_back(cur);
        cur = cur->next;
    }

    int i = 0, j = nodesArray.size() - 1; // i=0,j=n
    while (i < j) {
        nodesArray[i]->next = nodesArray[j]; // L0 -> Ln
        i++; // L1

        if (i == j) { // 这里极容易忽略，第一次写千万别暴露，需仔细推理
            break;
        }

        nodesArray[j]->next = nodesArray[i];
        j--;
    }
    nodesArray[i]->next = nullptr;
}
ListNode* middleNode(ListNode* head) { // 找中间节点，偶数偏左
    ListNode* slow = head;
    ListNode* fast = head;
    while (fast->next != nullptr && fast->next->next != nullptr) {
        // 若要找中间偏右只需 while(fast != nullptr && fast->next != nullptr)
        slow = slow->next;
        fast = fast->next->next;
    }
    return slow;
}
// 各一个合并
//    输入：head1: 1 -> 11 -> 111   head2: 2 -> 22 -> 222
//    输出：head1:  1 -> 2 -> 11 -> 22 -> 111 -> 222
void mergeList(ListNode* head1, ListNode* head2) {
    ListNode* head1nxt;
    ListNode* head2nxt;
    while (head1 != nullptr && head2 != nullptr) {
        head1nxt = head1->next;
        head2nxt = head2->next;

        head1->next = head2; // 1->2
        head2->next = head1nxt; // 2->11 // 容易写错，可漏破绽

        head1 = head1nxt; // head1 = 11
        head2 = head2nxt; // head2 = 22
    }
}
void reorderList22(ListNode* head) {
    if (head == nullptr) {
        return;
    }
    ListNode* mid = middleNode(head); // 返回中间偏左
    ListNode* firstHalf = head;
    ListNode* lastHalf = mid->next;
    mid->next = nullptr;

    lastHalf = reverseList(lastHalf);
    mergeList(firstHalf, lastHalf);
}

// 8.5 奇偶链表
// 题目描述：
//     输入：L0 -> L1 -> L2 -> L3 -> L4
//     输出：L0 -> L2 -> L4 -> L1 -> L3
// 要求O(1)空间，O(n)时间
// 思路：奇偶两个指针指出各个奇偶数，然后将偶数链表接到奇数链表后
ListNode* oddEvenList(ListNode* head) {
    if(head == nullptr) return head;

    ListNode* evenHead = head->next; // 偶数节点链表
    ListNode* odd = head, *even = evenHead;

    while(even != nullptr && even->next != nullptr){
        odd->next = even->next; //
        even->next = odd->next->next;

        odd = odd->next;
        even = even->next;
    }
    odd->next = evenHead; // 把偶节点链表接在后面
    return head;
}

// 8.6 循环有序链表中插入值
ListNode* insertValueInSortedDList(ListNode* head, int insertVal) {
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



//  9. 调整数组使奇数位于偶数前面，双指针
vector<int> exchange(vector<int>& nums) {
    int i = 0, j = nums.size() - 1;
    while (i < j){
        while(i < j && nums[i]%2 == 1) i++;
        while(i < j && nums[j]%2 == 0) j--;
        swap(nums[i], nums[j]);
    }
    return nums;
}


// --------------- 测试部分 ---------------------------------------------------
void test_sort_method(){
//    int a[] = {3,2,5,8,4,7,6,   9,1};
//
//    int n = sizeof(a)/sizeof(a[0]);
////    // shell_sort(a, n);
//    heap_sort(a, n);
//    // quick_sort(a, n, 0, n-1);
////    mergeSort(a, n);
////
////    for(int i = 0; i < n; i++){
////        printf("%d", a[i]);
////        if(i!=n-1) printf(",");
////    }
//
//    vector<int> nums1(a, a+4);
//    nums1.resize(9);
//    vector<int> nums2(a+4, a+9);
//    merge(nums1,4, nums2,5);
//
//    for(auto item : nums1){
//        cout<<item<<",";
//    }
//    // cout<<partitionl(a, n, 0, 6)<<endl;

    List l1;
    int a1[] = {1,8,3,6,5,4,7,2};
    for(int i = 0; i < sizeof(a1)/sizeof(a1[0]); i++){
        l1.push_back(a1[i]);
    }
    l1.print();
    reorderListEvenOdd1(l1._head);
    l1.print();
}

// 324. 摆动排序 II
// 题目描述：将数组重新排列成 nums[0]<nums[1]>nums[2]<nums[3]
//  （1）直观的NlogN解法：将数组进行排序，然后从中间位置进行等分（如果数组长度为奇数，则将中间的元素分到前面），然后将两个数组进行穿插。
//  （2）快选+三数排序：
//      找到 nums 的中位数，这一步可以通过「快速选择」算法来做，时间复杂度为 O(n)，空间复杂度为 O(logn)，假设找到的中位数为 x；
//      根据 nums[i] 与 x 的大小关系，将 nums[i] 分为三类（小于/等于/大于），划分三类的操作可以采用「三数排序」的做法，复杂度为 O(n)。
//      这一步做完之后，我们的数组调整为：[a_1, a_2, a_3, ... , b_1, b_2, b_3, ... , c_1, c_2, c_3]，即分成「小于 x / 等于 x / 大于 x」三段。
//  构造：先放「奇数」下标，再放「偶数」下标，放置方向都是「从左到右」（即可下标从小到大进行放置），放置的值是则是「从大到小」。
//    到这一步之前，我们使用到的空间上界是 O(\log{n})O(logn)，如果对空间上界没有要求的话，我们可以简单对 nums 进行拷贝，
//    然后按照对应逻辑进行放置即可，但这样最终的空间复杂度为 O(n)（代码见 P2）；
//    如果不希望影响到原有的空间上界，我们需要额外通过「找规律/数学」的方式，找到原下标和目标下标的映射关系（函数 getIdx 中）。
//


#endif //DATASTRUCT_ALGORITHM_SORT_H



