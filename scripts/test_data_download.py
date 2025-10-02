from ib_insync import IB, Stock
ib = IB(); ib.connect('127.0.0.1', 7496, clientId=7, readonly=True)

c = Stock('IWDA', 'SMART', 'USD', primaryExchange='LSEETF')
print('Qualify:', ib.qualifyContracts(c))

# Or discover the exact listing if you’re unsure:
for m in ib.reqMatchingSymbols('IWDA'):
    print(m)