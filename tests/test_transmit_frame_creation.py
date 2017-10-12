import random

N_t = 4
K = 4
modSymbolList = [[1], [-1]]
indexList = list(range(1, N_t + 1))
signalVector = []

#for k in range(0, K):
#    mod_symbol = random.choice(symbolList)
#    antenna_index = random.choice(indexList)
#    signalVector.append([0] * (antenna_index - 1) + mod_symbol + [0] * (N_t - antenna_index))
#
#print(signalVector)

possibleSymbols =[]
for mod_symbol in modSymbolList:
    for antenna_index in indexList:
        possibleSymbols.append([0] * (antenna_index - 1) + mod_symbol + [0] * (N_t - antenna_index))

for k in range(0, K):
    signalVector.append(random.choice(possibleSymbols))
    
print(possibleSymbols)
print(signalVector)
