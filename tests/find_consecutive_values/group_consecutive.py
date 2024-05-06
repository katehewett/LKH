def group_consecutive(test_list):
    result = []
    i = 0
    while i < len(test_list):
        j = i
        while j < len(test_list) - 1 and test_list[j+1] == test_list[j]+1:
            j += 1
        result.append((test_list[i], test_list[j]))
        i = j + 1
    return result
 
# initialize list
test_list = [1, 2, 3, 6, 7, 8, 11, 12, 13]
 
# printing original list
print("The original list is : " + str(test_list))
 
# Consecutive elements grouping list
res = group_consecutive(test_list)
 
# printing result
print("Grouped list is : " + str(res))

r = res[0]
idx1 = test_list.index((r[1]))