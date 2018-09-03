import random 

random.seed(42)

nums = list(range(1,11))

random.shuffle(nums)

print(nums)

print('train dirs: ' + str(nums[:6]))
print('valid dirs: ' + str(nums[6:8]))
print('test dirs:  ' + str(nums[8:]))

