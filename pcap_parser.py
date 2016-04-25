import dpkt
import socket
import binascii

def read_pcap_hdr(file):
    
    f = open(file)
    buf = f.read(dpkt.pcap.FileHdr.__hdr_len__)
    fh = dpkt.pcap.LEFileHdr(buf)
    print('Link Layer Type: ', fh.linktype)
    
    f.close()
    
    return

def get_tsval(opts):
    
    option_list = dpkt.tcp.parse_opts(opts)
    #print 'Here is the list of tcp options:'
    for options in option_list:
        #print options
        if options[0] == 8:
            timestamp = '0x'
            for h in options[1]:
                tmp = str(hex(ord(h)))
                if len(tmp) == 3:
                    hexfrag = tmp[-1]
                    timestamp = timestamp + '0' + hexfrag
                else:
                    hexfrag = tmp[-2:]
                    timestamp = timestamp + hexfrag
            tsval = int(timestamp[0:10], 16)            
        
    #return
    return float(tsval)

def parse(file):
    total = 0
    
    tcp_ts = []
    sys_ts = []
    
    f = open(file)
    pcap = dpkt.pcap.Reader(f)
    
    for ts, data in pcap:
        
        if total == 700:
            break
        
        eth = dpkt.ethernet.Ethernet(data)
        if eth.type == 2048:

            ip = eth.data
            
            try:
                tcp = ip.data
            except:
                pass
            
            src_ip = socket.inet_ntoa(ip.src)
            if (src_ip == '172.20.157.44') and (ip.p == 6):
                
                if tcp.flags == 24:
                    
                    tcp_ts.append(get_tsval(tcp.opts))
                    sys_ts.append(ts)
            
                #total += 1    
        total += 1
        
    f.close()
    
    
    #print 'The TCP timestamps are: \n', tcp_ts
    #print 'The System timestamps are: \n', sys_ts
    #print 'The # of timestamps collected is: ', len(tcp_ts)
    
    
    return tcp_ts, sys_ts

# ############ #
# Main Program #
# ############ #

ts_file = 'Overnight Trials.pcap'

#read_pcap_hdr(ts_file)

tcp_times, sys_times = parse(ts_file)

fn = 'Pcap Parsed Times.txt'
f1 = open(fn, 'w')

f1.write('Timestamps from 172.20.157.44\n')
f1.write(str(tcp_times) + '\n')
f1.write(str(sys_times) + '\n')

f1.close()

