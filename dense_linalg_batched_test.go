package tensor

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// This file covers MatMulBatched. Every numerical expectation is derived by
// looping over the batches and multiplying 2D matrices explicitly, so the
// batched engine is validated against the per-batch semantics rather than a
// parallel re-implementation of the same stride logic.

// refBatchedMM computes the expected (B, m, n) result by reading both operands
// through their logical coordinates (via At), so it is correct for transposed
// and offset views as well as contiguous tensors. ba/bb are the operand batch
// sizes (1 means broadcast). Values are float64.
func refBatchedMM(a *Dense, ba, m, k int, b *Dense, bb, n int) []float64 {
	batches := ba
	if bb > batches {
		batches = bb
	}
	out := make([]float64, batches*m*n)
	at := func(t *Dense, bi, i, j int) float64 {
		v, err := t.At(bi, i, j)
		if err != nil {
			panic(err)
		}
		return v.(float64)
	}
	for bi := 0; bi < batches; bi++ {
		ai, bj := bi, bi
		if ba == 1 {
			ai = 0
		}
		if bb == 1 {
			bj = 0
		}
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				var s float64
				for p := 0; p < k; p++ {
					s += at(a, ai, i, p) * at(b, bj, p, j)
				}
				out[bi*m*n+i*n+j] = s
			}
		}
	}
	return out
}

// refBatchedMM32 is the float32 counterpart.
func refBatchedMM32(a *Dense, ba, m, k int, b *Dense, bb, n int) []float32 {
	batches := ba
	if bb > batches {
		batches = bb
	}
	out := make([]float32, batches*m*n)
	at := func(t *Dense, bi, i, j int) float32 {
		v, err := t.At(bi, i, j)
		if err != nil {
			panic(err)
		}
		return v.(float32)
	}
	for bi := 0; bi < batches; bi++ {
		ai, bj := bi, bi
		if ba == 1 {
			ai = 0
		}
		if bb == 1 {
			bj = 0
		}
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				var s float32
				for p := 0; p < k; p++ {
					s += at(a, ai, i, p) * at(b, bj, p, j)
				}
				out[bi*m*n+i*n+j] = s
			}
		}
	}
	return out
}

// batchedCase is one row of the table-driven suite.
type batchedCase struct {
	name string
	// factory builds the two operands fresh for each run.
	makeA   func() *Dense
	makeB   func() *Dense
	ba      int // operand batch sizes (1 == broadcast)
	bb      int
	m, k, n int
	dtype   Dtype
	wantErr bool
}

func f64Backing(size, seed int) []float64 {
	v := make([]float64, size)
	for i := range v {
		v[i] = float64(seed + i)
	}
	return v
}

func f32Backing(size, seed int) []float32 {
	v := make([]float32, size)
	for i := range v {
		v[i] = float32(seed + i)
	}
	return v
}

func TestDense_MatMulBatched_Table(t *testing.T) {
	assert := assert.New(t)
	const B, M, K, N = 4, 3, 5, 2

	cases := []batchedCase{
		{
			name: "matching batches float64",
			ba:   B, bb: B, m: M, k: K, n: N, dtype: Float64,
			makeA: func() *Dense {
				return New(WithShape(B, M, K), WithBacking(f64Backing(B*M*K, 0)))
			},
			makeB: func() *Dense {
				return New(WithShape(B, K, N), WithBacking(f64Backing(B*K*N, 1)))
			},
		},
		{
			name: "matching batches float32",
			ba:   B, bb: B, m: M, k: K, n: N, dtype: Float32,
			makeA: func() *Dense {
				return New(WithShape(B, M, K), WithBacking(f32Backing(B*M*K, 0)))
			},
			makeB: func() *Dense {
				return New(WithShape(B, K, N), WithBacking(f32Backing(B*K*N, 1)))
			},
		},
		{
			name: "broadcast a batch=1 float64",
			ba:   1, bb: B, m: M, k: K, n: N, dtype: Float64,
			makeA: func() *Dense {
				return New(WithShape(1, M, K), WithBacking(f64Backing(M*K, 0)))
			},
			makeB: func() *Dense {
				return New(WithShape(B, K, N), WithBacking(f64Backing(B*K*N, 1)))
			},
		},
		{
			name: "broadcast b batch=1 float64",
			ba:   B, bb: 1, m: M, k: K, n: N, dtype: Float64,
			makeA: func() *Dense {
				return New(WithShape(B, M, K), WithBacking(f64Backing(B*M*K, 0)))
			},
			makeB: func() *Dense {
				return New(WithShape(1, K, N), WithBacking(f64Backing(K*N, 1)))
			},
		},
		{
			name: "single batch degenerate float64",
			ba:   1, bb: 1, m: M, k: K, n: N, dtype: Float64,
			makeA: func() *Dense {
				return New(WithShape(1, M, K), WithBacking(f64Backing(M*K, 0)))
			},
			makeB: func() *Dense {
				return New(WithShape(1, K, N), WithBacking(f64Backing(K*N, 1)))
			},
		},
		{
			name: "shared dimension k=1 float64",
			ba:   B, bb: B, m: M, k: 1, n: N, dtype: Float64,
			makeA: func() *Dense {
				return New(WithShape(B, M, 1), WithBacking(f64Backing(B*M, 0)))
			},
			makeB: func() *Dense {
				return New(WithShape(B, 1, N), WithBacking(f64Backing(B*N, 1)))
			},
		},
		{
			name: "transposed view of weights float64",
			ba:   B, bb: B, m: M, k: K, n: N, dtype: Float64,
			makeA: func() *Dense {
				return New(WithShape(B, M, K), WithBacking(f64Backing(B*M*K, 0)))
			},
			// Physical (B, N, K), logically transposed to (B, K, N).
			makeB: func() *Dense {
				w := New(WithShape(B, N, K), WithBacking(f64Backing(B*N*K, 2)))
				tt := w.Clone().(*Dense)
				require.NoError(t, tt.T(0, 2, 1))
				return tt
			},
		},
		{
			name: "offset view into larger slab float64",
			ba:   2, bb: 2, m: M, k: K, n: N, dtype: Float64,
			makeA: func() *Dense {
				big := New(WithShape(5, M, K), WithBacking(f64Backing(5*M*K, 0)))
				v, err := big.Slice(S(1, 3, 1)) // (2, M, K), offset into big
				require.NoError(t, err)
				return v.(*Dense)
			},
			makeB: func() *Dense {
				big := New(WithShape(5, K, N), WithBacking(f64Backing(5*K*N, 1)))
				v, err := big.Slice(S(2, 4, 1))
				require.NoError(t, err)
				return v.(*Dense)
			},
		},
		{
			name: "transposed offset view float64",
			ba:   2, bb: 2, m: M, k: K, n: N, dtype: Float64,
			makeA: func() *Dense {
				return New(WithShape(2, M, K), WithBacking(f64Backing(2*M*K, 0)))
			},
			makeB: func() *Dense {
				big := New(WithShape(4, N, K), WithBacking(f64Backing(4*N*K, 2)))
				v, err := big.Slice(S(1, 3, 1)) // (2, N, K) offset view
				require.NoError(t, err)
				tt := v.(*Dense).Clone().(*Dense)
				require.NoError(t, tt.T(0, 2, 1)) // -> (2, K, N)
				return tt
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a := tc.makeA()
			b := tc.makeB()

			res, err := a.MatMulBatched(b)
			if tc.wantErr {
				assert.Error(err)
				return
			}
			require.NoError(t, err)

			batches := tc.ba
			if tc.bb > batches {
				batches = tc.bb
			}
			assert.Equal(Shape{batches, tc.m, tc.n}, res.Shape())
			assert.True(res.DataOrder().IsRowMajor())

			switch tc.dtype {
			case Float64:
				want := refBatchedMM(a, tc.ba, tc.m, tc.k, b, tc.bb, tc.n)
				assert.Equal(want, res.Float64s(), "numerical mismatch")
			case Float32:
				want := refBatchedMM32(a, tc.ba, tc.m, tc.k, b, tc.bb, tc.n)
				assert.Equal(want, res.Float32s(), "numerical mismatch")
			}
		})
	}
}

// TestDense_MatMulBatched_EmptyAndDegenerate covers zero-sized batches and
// shared dimensions of zero or one.
func TestDense_MatMulBatched_EmptyAndDegenerate(t *testing.T) {
	assert := assert.New(t)

	// zero batch: result must be an empty (0, m, n) tensor and must not error.
	a := NewDense(Float64, Shape{0, 2, 3})
	b := NewDense(Float64, Shape{0, 3, 4})
	res, err := a.MatMulBatched(b)
	require.NoError(t, err)
	assert.Equal(Shape{0, 2, 4}, res.Shape())
	assert.Equal(0, len(res.Float64s()))

	// zero batch broadcast against a size-1 operand: a size-1 dimension
	// broadcasts to size 0, so the result is an empty (0, 2, 4) tensor. This
	// matches numpy's matmul broadcasting rather than raising an error.
	a0 := NewDense(Float64, Shape{0, 2, 3})
	b1 := NewDense(Float64, Shape{1, 3, 4})
	r0, err := a0.MatMulBatched(b1)
	require.NoError(t, err)
	assert.Equal(Shape{0, 2, 4}, r0.Shape())
	assert.Equal(0, len(r0.Float64s()))

	// the symmetric case: size-1 operand against a zero batch.
	a1 := NewDense(Float64, Shape{1, 2, 3})
	b0 := NewDense(Float64, Shape{0, 3, 4})
	r0b, err := a1.MatMulBatched(b0)
	require.NoError(t, err)
	assert.Equal(Shape{0, 2, 4}, r0b.Shape())

	// shared dimension k = 0: every result element must be exactly 0, matching
	// the per-batch 2D semantics (empty dot product sums to zero).
	az := NewDense(Float64, Shape{2, 3, 0})
	bz := NewDense(Float64, Shape{2, 0, 4})
	rz, err := az.MatMulBatched(bz)
	require.NoError(t, err)
	assert.Equal(Shape{2, 3, 4}, rz.Shape())
	for _, v := range rz.Float64s() {
		assert.Equal(0.0, v)
	}

	// m = 1 and n = 1 degenerate matrices.
	am := New(WithShape(2, 1, 3), WithBacking(f64Backing(6, 1)))
	bm := New(WithShape(2, 3, 1), WithBacking(f64Backing(6, 2)))
	rm, err := am.MatMulBatched(bm)
	require.NoError(t, err)
	want := refBatchedMM(am, 2, 1, 3, bm, 2, 1)
	assert.Equal(want, rm.Float64s())
}

// TestDense_MatMulBatched_InputsNotModified proves views and broadcast
// operands are addressed in place and never copied or overwritten.
func TestDense_MatMulBatched_InputsNotModified(t *testing.T) {
	assert := assert.New(t)
	const B, M, K, N = 3, 4, 5, 2

	// transposed view backed by an untouched slab
	w := New(WithShape(B, N, K), WithBacking(f64Backing(B*N*K, 7)))
	wBefore := append([]float64(nil), w.Float64s()...)
	wt := w.Clone().(*Dense)
	require.NoError(t, wt.T(0, 2, 1))

	a := New(WithShape(B, M, K), WithBacking(f64Backing(B*M*K, 0)))
	aBefore := append([]float64(nil), a.Float64s()...)

	// broadcast operand (batch 1)
	bb := New(WithShape(1, M, K), WithBacking(f64Backing(M*K, 3)))
	bbBefore := append([]float64(nil), bb.Float64s()...)

	_, err := a.MatMulBatched(wt)
	require.NoError(t, err)
	assert.Equal(wBefore, w.Float64s(), "transposed-view backing memory was modified")
	assert.Equal(aBefore, a.Float64s(), "operand a was modified")

	_, err = bb.MatMulBatched(New(WithShape(B, K, N), WithBacking(f64Backing(B*K*N, 1))))
	require.NoError(t, err)
	assert.Equal(bbBefore, bb.Float64s(), "broadcast operand was modified")
}

// TestDense_MatMulBatched_NonFinite ensures NaN/Inf propagate exactly as they
// do in a per-batch 2D multiplication.
func TestDense_MatMulBatched_NonFinite(t *testing.T) {
	assert := assert.New(t)
	const B, M, K, N = 2, 2, 2, 2
	av := []float64{1, math.NaN(), 3, math.Inf(1), 5, 6, 7, 8}
	bv := f64Backing(B*K*N, 1)
	a := New(WithShape(B, M, K), WithBacking(av))
	b := New(WithShape(B, K, N), WithBacking(bv))
	res, err := a.MatMulBatched(b)
	require.NoError(t, err)

	want := refBatchedMM(a, B, M, K, b, B, N)
	for i := range want {
		g, w := res.Float64s()[i], want[i]
		if math.IsNaN(w) {
			assert.True(math.IsNaN(g), "expected NaN at element %d, got %v", i, g)
			continue
		}
		assert.Equal(w, g, "non-finite propagation mismatch at %d", i)
	}
}

// TestDense_MatMulBatched_Deterministic verifies that broadcasting changes
// only addressing, not the per-batch accumulation order: the same operands
// produce bit-identical results across repeated calls and across the
// broadcast/non-broadcast formulations.
func TestDense_MatMulBatched_Deterministic(t *testing.T) {
	assert := assert.New(t)
	const B, M, K, N = 4, 3, 6, 3
	a := New(WithShape(B, M, K), WithBacking(f64Backing(B*M*K, 1)))
	b := New(WithShape(B, K, N), WithBacking(f64Backing(B*K*N, 2)))

	first, err := a.MatMulBatched(b)
	require.NoError(t, err)
	for i := 0; i < 3; i++ {
		again, err := a.MatMulBatched(b)
		require.NoError(t, err)
		assert.Equal(first.Float64s(), again.Float64s(), "results not bit-identical across calls")
	}

	// broadcast formulation: repeat b's first batch explicitly must equal the
	// batch=1 broadcast result element for element.
	b1 := New(WithShape(1, K, N), WithBacking(append([]float64(nil), b.Float64s()[:K*N]...)))
	bcast, err := a.MatMulBatched(b1)
	require.NoError(t, err)
	explicit, err := a.MatMulBatched(New(
		WithShape(B, K, N),
		WithBacking(repeatSlab(b.Float64s()[:K*N], B)),
	))
	require.NoError(t, err)
	assert.Equal(explicit.Float64s(), bcast.Float64s())
}

func repeatSlab(slab []float64, times int) []float64 {
	out := make([]float64, 0, len(slab)*times)
	for i := 0; i < times; i++ {
		out = append(out, slab...)
	}
	return out
}

// TestDense_MatMulBatched_ReuseAndIncr exercises buffer-reuse semantics,
// including that an incr target's existing values genuinely participate.
func TestDense_MatMulBatched_ReuseAndIncr(t *testing.T) {
	assert := assert.New(t)
	const B, M, K, N = 3, 2, 4, 3
	a := New(WithShape(B, M, K), WithBacking(f64Backing(B*M*K, 0)))
	b := New(WithShape(B, K, N), WithBacking(f64Backing(B*K*N, 1)))
	want := refBatchedMM(a, B, M, K, b, B, N)

	// plain
	res, err := a.MatMulBatched(b)
	require.NoError(t, err)
	assert.Equal(want, res.Float64s())

	// reuse: caller buffer is returned and overwritten.
	reuse := NewDense(Float64, Shape{B, M, N})
	rr, err := a.MatMulBatched(b, WithReuse(reuse))
	require.NoError(t, err)
	assert.True(rr == reuse, "reuse tensor was not returned")
	assert.Equal(want, rr.Float64s())

	// incr: existing values must be added, not zeroed.
	incr := NewDense(Float64, Shape{B, M, N})
	for i := range incr.Float64s() {
		incr.Float64s()[i] = 1000
	}
	ri, err := a.MatMulBatched(b, WithIncr(incr))
	require.NoError(t, err)
	assert.True(ri == incr, "incr tensor was not returned")
	for i := range want {
		assert.Equal(want[i]+1000, ri.Float64s()[i])
	}
}

// TestDense_MatMulBatched_ReuseViewOverlap covers a result buffer that is a
// view (offset into a larger allocation), distinct from the inputs.
func TestDense_MatMulBatched_ReuseViewOverlap(t *testing.T) {
	assert := assert.New(t)
	const B, M, K, N = 2, 2, 3, 2
	a := New(WithShape(B, M, K), WithBacking(f64Backing(B*M*K, 0)))
	b := New(WithShape(B, K, N), WithBacking(f64Backing(B*K*N, 1)))

	// a reuse view with an offset: backing is larger than the result.
	slab := NewDense(Float64, Shape{B + 2, M, N})
	view, err := slab.Slice(S(1, 1+B, 1))
	require.NoError(t, err)
	reuseView := view.(*Dense)
	assert.True(reuseView.IsView())

	rr, err := a.MatMulBatched(b, WithReuse(reuseView))
	require.NoError(t, err)
	want := refBatchedMM(a, B, M, K, b, B, N)
	assert.Equal(want, rr.Float64s())
}

// TestDense_MatMulBatched_InPlaceOverlapRejected ensures a result sharing
// memory with an input is refused rather than producing undefined BLAS
// behavior.
func TestDense_MatMulBatched_InPlaceOverlapRejected(t *testing.T) {
	assert := assert.New(t)
	const B, M, K = 2, 2, 3
	// square (B,M,M) so the output can alias an input.
	a := New(WithShape(B, M, M), WithBacking(f64Backing(B*M*M, 0)))
	b := New(WithShape(B, M, M), WithBacking(f64Backing(B*M*M, 5)))

	// reuse buffer backed by the same memory as a.
	alias := a.ShallowClone()
	alias.setShape(B, M, M)
	_, err := a.MatMulBatched(b, WithReuse(alias))
	assert.Error(err)

	// incr alias is rejected as well.
	_, err = a.MatMulBatched(b, WithIncr(a.ShallowClone()))
	assert.Error(err)
}

// TestDense_MatMulBatched_NonBroadcastable covers failure path 1.
func TestDense_MatMulBatched_NonBroadcastable(t *testing.T) {
	assert := assert.New(t)
	a := New(WithShape(3, 2, 2), WithBacking(f64Backing(12, 0)))
	b := New(WithShape(2, 2, 2), WithBacking(f64Backing(8, 1)))
	_, err := a.MatMulBatched(b)
	require.Error(t, err)
	// both batch shapes must be reported.
	msg := err.Error()
	assert.Contains(msg, "(3)")
	assert.Contains(msg, "(2)")
}

// TestDense_MatMulBatched_IllegalView covers failure path 2: a view BLAS
// cannot address must be rejected, never silently copied.
func TestDense_MatMulBatched_IllegalView(t *testing.T) {
	assert := assert.New(t)
	const B, M, K, N = 2, 3, 4, 2
	base := New(WithShape(B, M, K), WithBacking(f64Backing(B*M*K, 0)))
	before := append([]float64(nil), base.Float64s()...)

	// step-2 slice on the inner dimension -> unsupported inner stride.
	v, err := base.Slice(nil, nil, S(0, K, 2))
	require.NoError(t, err)
	b := New(WithShape(B, 2, N), WithBacking(f64Backing(B*2*N, 1)))
	_, err = v.(*Dense).MatMulBatched(b)
	require.Error(t, err)
	assert.Contains(err.Error(), "cannot be addressed in place")

	// the original memory must be untouched: no silent materialization.
	assert.Equal(before, base.Float64s())

	// col-major batched tensors are likewise rejected, not reordered. A
	// col-major output buffer keeps all three operands col-major so the call
	// reaches the interleaving check rather than the order-mismatch check.
	af := New(WithShape(B, M, K), AsFortran(f64Backing(B*M*K, 0)))
	bf := New(WithShape(B, K, N), AsFortran(f64Backing(B*K*N, 1)))
	cf := NewDense(Float64, Shape{B, M, N}, AsFortran(nil))
	_, err = af.MatMulBatched(bf, WithReuse(cf))
	require.Error(t, err)
	assert.Contains(err.Error(), "col-major")
}

// TestDense_MatMulBatched_NegativeStride constructs a tensor whose inner
// stride is negative and confirms it is rejected rather than copied.
func TestDense_MatMulBatched_NegativeStride(t *testing.T) {
	assert := assert.New(t)
	const B, M, K, N = 1, 2, 3, 2
	src := New(WithShape(B, M, K), WithBacking(f64Backing(B*M*K, 0)))
	neg := src.ShallowClone()
	neg.setShape(B, M, K)
	neg.AP = MakeAP(Shape{B, M, K}, []int{M * K, K, -1}, MakeDataOrder(NonContiguous), 0)

	b := New(WithShape(B, K, N), WithBacking(f64Backing(B*K*N, 1)))
	_, err := neg.MatMulBatched(b)
	require.Error(t, err)
	assert.Contains(err.Error(), "negative stride")
}

// TestDense_MatMulBatched_OutOfBounds constructs a view whose non-negative
// strides would address beyond its backing memory and confirms the bounds
// check rejects it instead of reading out of range or panicking in BLAS.
func TestDense_MatMulBatched_OutOfBounds(t *testing.T) {
	assert := assert.New(t)
	const B, M, K, N = 2, 2, 3, 2
	src := New(WithShape(B, M, K), WithBacking(f64Backing(B*M*K, 0)))
	oob := src.ShallowClone()
	oob.setShape(B, M, K)
	// Standard (K, 1) inner strides but an oversized batch stride: the last
	// batch would walk past the backing array.
	oob.AP = MakeAP(Shape{B, M, K}, []int{M * K * 3, K, 1}, MakeDataOrder(NonContiguous), 0)

	b := New(WithShape(B, K, N), WithBacking(f64Backing(B*K*N, 1)))
	_, err := oob.MatMulBatched(b)
	require.Error(t, err)
	assert.Contains(err.Error(), "backing memory only holds")
}

// TestDense_MatMulBatched_ScratchLimit covers failure path 3.
func TestDense_MatMulBatched_ScratchLimit(t *testing.T) {
	assert := assert.New(t)
	const B, M, K, N = 8, 16, 16, 16
	a := New(WithShape(B, M, K), WithBacking(f64Backing(B*M*K, 0)))
	b := New(WithShape(B, K, N), WithBacking(f64Backing(B*K*N, 1)))

	orig := BatchedMatMulMaxScratch()
	defer SetBatchedMatMulMaxScratch(orig)

	// an impossibly small limit must trigger a refusal naming both numbers.
	SetBatchedMatMulMaxScratch(1)
	_, err := a.MatMulBatched(b)
	require.Error(t, err)
	msg := err.Error()
	assert.Contains(msg, "exceeds the limit")
	assert.Contains(msg, "1 bytes")

	// restoring the default limit makes the same call succeed.
	SetBatchedMatMulMaxScratch(0)
	res, err := a.MatMulBatched(b)
	require.NoError(t, err)
	want := refBatchedMM(a, B, M, K, b, B, N)
	assert.Equal(want, res.Float64s())
}

// TestDense_MatMulBatched_AllocationsDoNotScaleWithBatches is the performance
// contract: addressing views in place means the number of heap allocations per
// call must stay constant as the batch count grows. A reused output buffer
// drives the steady-state allocations to zero.
func TestDense_MatMulBatched_AllocationsDoNotScaleWithBatches(t *testing.T) {
	const M, K, N = 4, 4, 4
	measure := func(batches int) float64 {
		a := New(WithShape(batches, M, K), WithBacking(f64Backing(batches*M*K, 0)))
		b := New(WithShape(batches, K, N), WithBacking(f64Backing(batches*K*N, 1)))
		reuse := NewDense(Float64, Shape{batches, M, N})
		return testing.AllocsPerRun(20, func() {
			_, err := a.MatMulBatched(b, WithReuse(reuse))
			if err != nil {
				t.Fatal(err)
			}
		})
	}

	small := measure(2)
	large := measure(64)
	t.Logf("allocs/run: batches=2 -> %v, batches=64 -> %v", small, large)
	// allocations must not grow with batch count; allow a tiny constant floor.
	assert.LessOrEqual(t, large, small+0.0,
		"allocations scaled with batch count (small=%v large=%v)", small, large)
}

// TestDense_MatMulBatched_BadShapesAndTypes guards the entry-level validation.
func TestDense_MatMulBatched_BadShapesAndTypes(t *testing.T) {
	assert := assert.New(t)

	// 2D operands are still handled exclusively by the existing MatMul path.
	two := New(WithShape(2, 3), WithBacking(f64Backing(6, 0)))
	three := New(WithShape(2, 3, 2), WithBacking(f64Backing(12, 0)))
	_, err := two.MatMulBatched(three)
	assert.Error(err)

	// mismatched shared dimension
	a := New(WithShape(2, 3, 4), WithBacking(f64Backing(24, 0)))
	b := New(WithShape(2, 5, 2), WithBacking(f64Backing(20, 1)))
	_, err = a.MatMulBatched(b)
	assert.Error(err)

	// unsupported dtype
	ai := New(WithShape(2, 3, 4), WithBacking(Range(Int, 0, 24)))
	bi := New(WithShape(2, 4, 2), WithBacking(Range(Int, 0, 16)))
	_, err = ai.MatMulBatched(bi)
	assert.Error(err)
}
