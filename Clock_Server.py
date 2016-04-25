import socket
import sys
import time


HOST = sys.argv[1]
PORT = 25000

# Create a TCP/IP socket
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the address given on the command line
server_address = (HOST, PORT)
print ('starting up on %s port %s' % server_address)
server.bind(server_address)

# Awaiting a connection request from a client
server.listen(1)
while True:
    client, address = server.accept()
    
    #client.sendall(b'Welcome!  How many timestamps would you like to take?')
    loops = int(client.recv(4096))
    
    clock_array = []
    for i in range(loops):
        VM_clock = time.time() # Wall clock time of the VM
        client.sendall(str(VM_clock).encode())
        clock_array.append(VM_clock)
        time.sleep(1)
    
    #print 'The times sent were:', clock_array
    
    client.shutdown(0)
    client.close()
    