package main

import (
	"crypto/sha256"
	"bufio"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"os"
	"time"
	"encoding/binary"
    "shufflemessage/modp"
	"shufflemessage/mycrypto"
)

type Req struct {
    D             int      `json:"D"`
    K             int      `json:"k"`
    TauCount      int      `json:"tau_count"`
    Mode          string   `json:"mode"`            
    RowsB64       []string `json:"rows_b64"`        
    STrustBitsB64 string   `json:"s_trust_bits_b64"` 
    // MPC
    STrustShareB64 string   `json:"s_trust_share_b64"`
    SeedsB64       []string `json:"seeds_b64"`      
    Role           int      `json:"role"`            
    Leader         bool     `json:"leader"`
    ListenAddr     string   `json:"listen_addr"`    
    PeerAddr       string   `json:"peer_addr"`        
	RoundID        int      `json:"round_id"` 
}


type Resp struct {
	PassMask []int  `json:"pass_mask,omitempty"`
	HD       []int  `json:"hd,omitempty"` 
	Msg      string `json:"msg,omitempty"`
}

func main() {
	raw, err := io.ReadAll(bufio.NewReader(os.Stdin))
	if err != nil { fail(err); return }
	var req Req
	if err := json.Unmarshal(raw, &req); err != nil { fail(err); return }

	rows := decodeRows(req.RowsB64)

	switch req.Mode {
	case "plain":
		sBits := mustB64(req.STrustBitsB64)
		hd, err := mycrypto.HammingPlain(rows, req.D, req.K, sBits)
		if err != nil { fail(err); return }
		pm := make([]int, len(hd))
		for i, v := range hd {
			if v <= req.TauCount { pm[i] = 1 }
		}
		ok(Resp{PassMask: pm, HD: hd})

	case "mpc":
	    sShare := mustB64(req.STrustShareB64)
	    rows := decodeRows(req.RowsB64)
	    hdShare, err := mycrypto.HammingSharesSim(rows, req.D, req.K, sShare)
	    if err != nil { fail(err); return }

	    hdVals := make([]int, len(rows))
	    for i := 0; i < len(rows); i++ {
	        var e modp.Element
	        e.SetBytes(hdShare[16*i : 16*(i+1)])
	        hdVals[i] = int(binary.LittleEndian.Uint64(e.Bytes()[:8]))
	    }

	    pm := make([]int, len(rows))
	    for i, v := range hdVals {
	        if v <= req.TauCount { pm[i] = 1 }
	    }
	    ok(Resp{PassMask: pm, HD: hdVals})
		

	case "shuffle":

    	salt := os.Getenv("RAIN_SHUFFLE_SALT") 
    	seedMaterial := fmt.Sprintf("perm|round=%d|salt=%s", req.RoundID, salt)
    	h := sha256.Sum256([]byte(seedMaterial))
    	seed16 := make([]byte, 16)
    	copy(seed16, h[:16])

    	perm := mycrypto.GenPerm(req.D, seed16)

    	if len(req.RowsB64) == 0 {
    	    b, _ := json.Marshal(struct {
    	        Perm []int `json:"perm"`
    	    }{Perm: perm})
    	    fmt.Println(string(b))
    	    return
    	}

   
    	flat, _ := base64.StdEncoding.DecodeString(req.RowsB64[0])
    	shuffled, perm2, err := mycrypto.SimulateSecureShuffle(flat, req.D, req.K)
    	if err != nil { fail(err); return }

    	resp := struct {
    	    ShuffledB64 string `json:"shuffled_b64"`
    	    Perm        []int  `json:"perm"`
    	}{
    	    ShuffledB64: base64.StdEncoding.EncodeToString(shuffled),
    	    Perm:        perm2,
    	}
    	b, _ := json.Marshal(resp)
    	fmt.Println(string(b))


	default:
		fail(fmtErr("unsupported mode: %s", req.Mode))
	}
}

func ok(r Resp){ b,_:=json.Marshal(r); fmt.Println(string(b)) }
func fail(err error){ b,_:=json.Marshal(Resp{Msg:err.Error()}); fmt.Println(string(b)); os.Exit(1) }

func decodeRows(list []string) [][]byte {
	out := make([][]byte, len(list))
	for i, s := range list {
		b, _ := base64.StdEncoding.DecodeString(s)
		out[i] = b
	}
	return out
}
func mustB64(s string) []byte {
	b, _ := base64.StdEncoding.DecodeString(s); return b
}
func fmtErr(f string, a ...interface{}) error { return fmt.Errorf(f, a...) }

func dialPeer(role int, listenAddr, peerAddr string) (net.Conn, error) {
	if role == 0 {
		ln, err := net.Listen("tcp", listenAddr)
		if err != nil { return nil, err }
		defer func(){ go func(){ time.Sleep(5*time.Second); ln.Close() }() }()
		conn, err := ln.Accept()
		return conn, err
	}
	// role == 1
	var conn net.Conn
	var err error
	for i:=0; i<50; i++ {
		conn, err = net.Dial("tcp", peerAddr)
		if err == nil { break }
		time.Sleep(100 * time.Millisecond)
		fmt.Fprintln(os.Stderr, "[rain_hd] role", role, "listening", listenAddr)
		fmt.Fprintln(os.Stderr, "[rain_hd] role", role, "connecting", peerAddr, "try", i)
	}
	

	return conn, err
}
func writeAll(c net.Conn, b []byte) error {
	n := 0
	for n < len(b) {
		m, err := c.Write(b[n:])
		if err != nil { return err }
		n += m
	}
	return nil
}
func readN(c net.Conn, n int) ([]byte, error) {
	buf := make([]byte, n)
	off := 0
	for off < n {
		m, err := c.Read(buf[off:])
		if err != nil { return nil, err }
		off += m
	}
	return buf, nil
}


func addShares(a, b []byte) []byte {
	out := make([]byte, len(a))
	copy(out, a)
	mycrypto.AddOrSub(out, b, true) // out += b
	return out
}
