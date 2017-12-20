import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)

# we can do :
#result = x1 * x2

#but tensorflow help in speeding computation, as python reads the bufffer line by line, tensor flow uses c
#  so we do

result = tf.multiply(x1,x2)
#it prints a tensor 
print(result)

##sess = tf.Session()
##print(sess.run(result))
##sess.close()
#better way to do it with with in python
with tf.Session() as sess :
    output = (sess.run(result))
    print(output)
print(output)
