#include <iostream>
#include <list>
#include "sort_methods.h"
#include "kuohao_match.h"
#include "list_op.h"
#include "two_sum.h"
#include "find_kth_largest.h"
#include "sliding_window.h"
// #include "binary_tree_inorder_traversal.h"
// #include "ip_address_op.h"
#include "binary_tree_serialize.h"
#include "dp_1d.h"
#include "dp_2d.h" // ""
#include "dp_2dm.h"
#include "array_rotate.h"
#include "array_duplicate.h"
#include "array_2d.h"
#include "calculater.h"
#include "bit_op.h"
#include "array_string_op.h"
#include "keChengBiao_tp.h"
#include "dp_hd.h"
#include "stack_op.h"
#include "array_presum.h"
#include "array_subset_divide.h"

using namespace std;
// 5000*  1万

bool isPalindrome(list<int> head);

int main() {
    std::cout << "Hello, World!" << std::endl;
   // testcanIWin();

    //int a[] = {3,2,5,8,4,7,6,9};
     //InsertSort(a,8);
    // SelectSort(a,8);
    // BubbleSort(a, 8);

     //test_sort_method();
//    for(auto i: a){
//        cout<<i<<" ";
//    }
//    cout<<endl;



    // test_maxEnvelopes();
//    if(isValid("()")){
//        cout<<"kuohao match success"<<endl;
//    }else{
//        cout<<"kuohao match fail"<<endl;
//    }

    // test_kuohao_match();
    // test_list_new();

    // test_three_sum();
     // testKlargest();
    // test_MedianFinder();

    // test_minwindow();
    // test_lenLSubstring();

    // test_inorderTraversal_stk();
    // test_valid_ip();

    // test_serialize();
    // test_numDecodings();
    // test_t1();

    // test();
    // test_calculate();
    // testPrintOrder();
    // testNumIslands();

    // testswap();
    // testappearOnlyOnceChar();
    // testreversePairs();

    // testfindUnsortedSubarray();

    // testmaxSubarraySumCircular();
    // testCanFinish();

    // test_sort_method();

    // testnumsTranslate();

    // testcoinChange();
    // cout << removeDuplicateLetters("bcabc")<< endl;
    // test_findRepeatedDnaSequences();
    // test_findContinuousSequence();
    // test_minMeetingRooms();


    // test_validateStackSequences();

    // test_platesBetweenCandles();

    // test_distributeCookies();


    std::cout << "Hello, World!" << std::endl;
    return 0;
}



//bool isPalindrome(list<int> head) {
//    auto reversed = head;
//
//    for(auto p = head.begin(), q = reversed.begin(); p != head.end(); p++, q++){
//        if(*p != *q){
//            return false;
//        }
//    }
//    return true;
//}

// 长度为intLength的第k小回文数 https://leetcode-cn.com/contest/weekly-contest-286/ranking/ t3
vector<long long> kthPalindrome(vector<int>& queries, int intLength) {
    std::vector<long long> ans;
    for (auto x : queries) {
        int k = (intLength + 1) / 2;
        int p = 1;
        for (int i = 0; i < k - 1; i++) {
            p *= 10;
        }
        if (x > 9 * p) {
            ans.push_back(-1);
        } else {
            long long res = p + x - 1;
            auto s = std::to_string(res);
            s.resize(intLength - k);
            std::reverse(s.begin(), s.end());
            for (auto c : s) {
                res = 10 * res + c - '0';
            }
            ans.push_back(res);
        }
    }
    return ans;
}
