import socket
import sys
import time


def time_collect(SERVER, loops):
    
    PORT = 25000

    # Request connection to server
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (SERVER, PORT)

    print('\nconnecting to %s port %s\n' % server_address)

    client.connect(server_address) # Connect to server

    client.sendall(str(loops).encode())

    vm_array = []
    host_array = []

    for i in range(int(loops)):
        data = client.recv(4096).decode() # Data was converted to binary, decoding back to Unicode
        Host_clock = time.time() # Wall clock time of client
        VM_clock = float(data) # Times received as a string, convert back to float
        vm_array.append(VM_clock)
        host_array.append(Host_clock)

    #print('The timestamps of VM ', SERVER, ' were:', vm_array)
    #print('\nThe times the VM timestamps were received were:', host_array)
    
    #f.write('The timestamps of VM ' + str(SERVER) + ' were: ' + str(vm_array) + '\n')
    #f.write('The times the VM timestamps were received were: ' + str(host_array) + '\n')
    
    f.write(str(vm_array) + '\n')
    f.write(str(host_array) + '\n')    
    
    client.close()
        
    return

#*************************
# ***** MAIN PROGRAM *****
#*************************

samples = input('\nWelcome!  How many timestamps would you like to take from each VM? ')
trials = input('\nHow many trial runs would you like to do? ')

#f = open('Test2.txt', 'w')

for i in range(int(trials)):
    fn = 'Pcap_HW_and_VM_Trial_' + str(i+1) + '.txt'
    f = open(fn, 'w')
    f.write('Test run # ' + str(i+1) + '\n')
    
    servers = sys.argv[1:]
    #servers = [sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]]
    
    for i in range(len(servers)):
        time_collect(servers[i], samples)
    
    f.close()