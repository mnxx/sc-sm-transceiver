#import itertools
#
#def split_users(users_list):
#    users = set(users_list)
#    for comb in itertools.combinations(users, int(len(users)/2)):
#        control = set(comb)
#        treatment = users - control
#        yield control, treatment
#
#users = {"A", "B", "C", "D", "E", "F"}
#
#for control, treatment in split_users(users):
#    print ("Control" +  str(control) +  "treatment" + str(treatment))
#

floored_stuff = 3
N_t = 2**floored_stuff

spatialList = []
for i in range (0, N_t):
    # Format the decimal number to bits represented by a string.
    symbolString = "{0:b}".format(i)
    # Cast the string to an integer and split it in a list.
    symbol = list(map(int, symbolString))
    # Append the list as a symbol.
    spatialList.append(symbol)
    # Pad zeros in front of the symbols which are too short.
    while len(spatialList[i]) != floored_stuff:
        spatialList[i] = [0] + spatialList[i]

#print(spatialList)

mergedList = []
symbolList = [[1], [-1]]
for sym in symbolList:
    for spsym in spatialList:
        mergedList.append(sym + spsym)

print(mergedList)
