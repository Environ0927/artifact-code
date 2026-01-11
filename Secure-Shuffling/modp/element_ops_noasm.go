

package modp



import "math/bits"

func mul(z, x, y *Element) {
	_mulGeneric(z, x, y)
}

func square(z, x *Element) {
	_squareGeneric(z, x)
}

// FromMont converts z in place (i.e. mutates) from Montgomery to regular representation
// sets and returns z = z * 1
func fromMont(z *Element) {
	_fromMontGeneric(z)
}

func add(z, x, y *Element) {
	var carry uint64

	z[0], carry = bits.Add64(x[0], y[0], 0)
	z[1], carry = bits.Add64(x[1], y[1], carry)
	// if we overflowed the last addition, z >= q
	// if z >= q, z = z - q
	if carry != 0 {
		// we overflowed, so z >= q
		z[0], carry = bits.Sub64(z[0], 18446744073709551457, 0)
		z[1], carry = bits.Sub64(z[1], 18446744073709551615, carry)
		return
	}

	// if z > q --> z -= q
	// note: this is NOT constant time
	if !(z[1] < 18446744073709551615 || (z[1] == 18446744073709551615 && (z[0] < 18446744073709551457))) {
		var b uint64
		z[0], b = bits.Sub64(z[0], 18446744073709551457, 0)
		z[1], _ = bits.Sub64(z[1], 18446744073709551615, b)
	}
}

func double(z, x *Element) {
	var carry uint64

	z[0], carry = bits.Add64(x[0], x[0], 0)
	z[1], carry = bits.Add64(x[1], x[1], carry)
	// if we overflowed the last addition, z >= q
	// if z >= q, z = z - q
	if carry != 0 {
		// we overflowed, so z >= q
		z[0], carry = bits.Sub64(z[0], 18446744073709551457, 0)
		z[1], carry = bits.Sub64(z[1], 18446744073709551615, carry)
		return
	}

	// if z > q --> z -= q
	// note: this is NOT constant time
	if !(z[1] < 18446744073709551615 || (z[1] == 18446744073709551615 && (z[0] < 18446744073709551457))) {
		var b uint64
		z[0], b = bits.Sub64(z[0], 18446744073709551457, 0)
		z[1], _ = bits.Sub64(z[1], 18446744073709551615, b)
	}
}

func sub(z, x, y *Element) {
	var b uint64
	z[0], b = bits.Sub64(x[0], y[0], 0)
	z[1], b = bits.Sub64(x[1], y[1], b)
	if b != 0 {
		var c uint64
		z[0], c = bits.Add64(z[0], 18446744073709551457, 0)
		z[1], _ = bits.Add64(z[1], 18446744073709551615, c)
	}
}

func neg(z, x *Element) {
	if x.IsZero() {
		z.SetZero()
		return
	}
	var borrow uint64
	z[0], borrow = bits.Sub64(18446744073709551457, x[0], 0)
	z[1], _ = bits.Sub64(18446744073709551615, x[1], borrow)
}

func reduce(z *Element) {

	// if z > q --> z -= q
	// note: this is NOT constant time
	if !(z[1] < 18446744073709551615 || (z[1] == 18446744073709551615 && (z[0] < 18446744073709551457))) {
		var b uint64
		z[0], b = bits.Sub64(z[0], 18446744073709551457, 0)
		z[1], _ = bits.Sub64(z[1], 18446744073709551615, b)
	}
}
