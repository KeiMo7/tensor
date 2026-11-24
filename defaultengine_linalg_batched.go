package tensor

import (
	"unsafe"

	"github.com/pkg/errors"
	"gonum.org/v1/gonum/blas"
)

// This file implements batched matrix multiplication for the standard engine.
//
// Unlike a naive loop that slices each batch out into a new 2D *Dense and
// calls MatMul on it, this implementation addresses every batch directly in
// the caller's backing memory through the tensor strides. The inputs are never
// copied (not even when they are transposed views, offset views or views with
// non-default strides), and the hot path performs no heap allocations at all:
// one BLAS call per batch against re-sliced windows of the original arrays.

// batchedMatMulMaxScratchBytes bounds the working-set size of a single
// MatMulBatched call. The default engine needs no scratch memory of its own -
// the limit guards the per-batch BLAS windows and the output working set so
// that a pathological request fails fast with an explanatory error instead of
// driving the process into swapping or an out-of-memory condition.
//
// The value is expressed in bytes and may be overridden with
// SetBatchedMatMulMaxScratch.
var batchedMatMulMaxScratchBytes = 64 * 1024 * 1024

// BatchedMatMulMaxScratch returns the current per-call working-set limit for
// MatMulBatched, in bytes.
func BatchedMatMulMaxScratch() int { return batchedMatMulMaxScratchBytes }

// SetBatchedMatMulMaxScratch overrides the per-call working-set limit for
// MatMulBatched. A non-positive value restores the default limit.
func SetBatchedMatMulMaxScratch(bytes int) {
	if bytes <= 0 {
		batchedMatMulMaxScratchBytes = 64 * 1024 * 1024
		return
	}
	batchedMatMulMaxScratchBytes = bytes
}

// matLayout is the stride-level description of one batched operand. It is
// derived purely from a tensor's access pattern and contains no pointers, so
// building it allocates nothing on the heap.
type matLayout struct {
	rows, cols int // logical matrix dimensions for every batch
	// ld is the BLAS leading dimension: the stride (in elements) between
	// successive rows of the row-major per-batch matrix.
	ld int
	// rowStride and colStride are the element strides between adjacent rows
	// and columns of a single batch matrix.
	rowStride, colStride int
	// batchStride is the element stride between the same matrix element of
	// two successive batches. It is 0 for a broadcast operand.
	batchStride int
	// trans reports whether the logical matrix is transposed with respect to
	// its physical storage, i.e. whether BLAS must be handed blas.Trans.
	trans bool
}

// MatMulBatched performs a batched matrix multiplication for the standard
// engine. The result is written into prealloc; its initial contents are
// overwritten (beta = 0), exactly as StdEng.MatMul does for the 2D case.
func (e StdEng) MatMulBatched(a, b, prealloc Tensor) (err error) {
	ad, bd, pd, err := e.checkBatchedMatMulTensors(a, b, prealloc)
	if err != nil {
		return err
	}

	la, lb, lc, batches, err := buildBatchedMatMulLayouts(ad, bd, pd)
	if err != nil {
		return err
	}

	// The default engine allocates no scratch buffer; the working set the
	// limit protects is the memory touched per call: the full output plus the
	// two largest per-batch BLAS windows.
	elemSize := int(ad.rtype().Size())
	workingSet := batches*lc.rows*lc.cols + la.rows*la.cols + lb.rows*lb.cols
	if need := workingSet * elemSize; need > batchedMatMulMaxScratchBytes {
		return errors.Errorf(
			"MatMulBatched: required working set of %d bytes exceeds the limit of %d bytes "+
				"(batches=%d, matrix=%dx%d, shared=%d, dtype size=%d)",
			need, batchedMatMulMaxScratchBytes, batches, la.rows, lb.cols, la.cols, elemSize,
		)
	}

	// Recover from any BLAS panic (defensive: the layout validation above is
	// meant to make them unreachable) and turn it into a regular error so a
	// bad view can never crash the caller.
	defer func() {
		if r := recover(); r != nil {
			err = errors.Errorf("MatMulBatched: BLAS rejected the strided view: %v", r)
		}
	}()

	// A zero-sized output (empty batch, or an empty result matrix) needs no
	// arithmetic; its backing array may legitimately be nil. The per-batch 2D
	// semantics produce nothing in that case.
	if batches == 0 || lc.rows == 0 || lc.cols == 0 {
		return nil
	}
	// A zero shared dimension means every dot product is the empty sum, i.e.
	// zero. The inputs may have nil backing here; mirroring GEMM's beta = 0
	// rule, the correct result is simply a zeroed output.
	if la.cols == 0 {
		pd.Zero()
		return nil
	}

	switch av := ad.Data().(type) {
	case []float64:
		bv := bd.Float64s()
		cv := pd.Float64s()
		batchedDgemm(la, lb, lc, batches, av, bv, cv)
	case []float32:
		bv := bd.Float32s()
		cv := pd.Float32s()
		batchedSgemm(la, lb, lc, batches, av, bv, cv)
	default:
		return errors.Errorf(typeNYI, "MatMulBatched", ad.Data())
	}
	return nil
}

// checkBatchedMatMulTensors performs the engine-level checks that mirror
// checkThreeFloatComplexTensors for the 2D path, restricted to the two real
// floating point types the batched path supports.
func (e StdEng) checkBatchedMatMulTensors(a, b, ret Tensor) (ad, bd, pd DenseTensor, err error) {
	if err = e.checkAccessible(a); err != nil {
		return nil, nil, nil, errors.Wrap(err, "checkBatchedMatMul: a is not accessible")
	}
	if err = e.checkAccessible(b); err != nil {
		return nil, nil, nil, errors.Wrap(err, "checkBatchedMatMul: b is not accessible")
	}
	if err = e.checkAccessible(ret); err != nil {
		return nil, nil, nil, errors.Wrap(err, "checkBatchedMatMul: ret is not accessible")
	}

	if a.Dtype() != b.Dtype() || b.Dtype() != ret.Dtype() {
		return nil, nil, nil, errors.New("Expected a and b and retVal all to have the same Dtype")
	}

	switch a.Dtype() {
	case Float32, Float64:
	default:
		return nil, nil, nil, errors.Errorf("MatMulBatched only supports Float32 and Float64. Got %v instead", a.Dtype())
	}

	if ad, err = getDenseTensor(a); err != nil {
		return nil, nil, nil, errors.Wrap(err, "checkBatchedMatMul expects a to be a DenseTensor")
	}
	if bd, err = getDenseTensor(b); err != nil {
		return nil, nil, nil, errors.Wrap(err, "checkBatchedMatMul expects b to be a DenseTensor")
	}
	if pd, err = getDenseTensor(ret); err != nil {
		return nil, nil, nil, errors.Wrap(err, "checkBatchedMatMul expects ret to be a DenseTensor")
	}
	return ad, bd, pd, nil
}

// buildBatchedMatMulLayouts validates the shapes, strides and memory bounds of
// the three operands and derives their BLAS layouts, together with the
// (post-broadcast) batch count.
func buildBatchedMatMulLayouts(a, b, c DenseTensor) (la, lb, lc matLayout, batches int, err error) {
	if a.Dims() < 3 || b.Dims() < 3 || c.Dims() < 3 {
		return la, lb, lc, 0, errors.Errorf(
			"MatMulBatched requires operands with at least 3 dimensions (batch + matrix). "+
				"Got a: %v, b: %v, ret: %v", a.Shape(), b.Shape(), c.Shape())
	}

	batchA := a.Shape()[0]
	batchB := b.Shape()[0]
	batchC := c.Shape()[0]

	// broadcast rules: equal batch sizes, or one side of size 1.
	switch {
	case batchA == batchB:
		batches = batchA
	case batchA == 1:
		batches = batchB
	case batchB == 1:
		batches = batchA
	default:
		return la, lb, lc, 0, errors.Errorf(
			"MatMulBatched: batch dimensions cannot be broadcast: a's batch shape is %v, b's batch shape is %v",
			a.Shape()[:1], b.Shape()[:1])
	}
	if batchC != batches {
		return la, lb, lc, 0, errors.Errorf(
			"MatMulBatched: result batch size %d does not match the broadcast batch size %d", batchC, batches)
	}

	m, k1 := a.Shape()[a.Dims()-2], a.Shape()[a.Dims()-1]
	k2, n := b.Shape()[b.Dims()-2], b.Shape()[b.Dims()-1]
	if k1 != k2 {
		return la, lb, lc, 0, errors.Errorf(shapeMismatch, a.Shape(), b.Shape())
	}
	if cm, cn := c.Shape()[c.Dims()-2], c.Shape()[c.Dims()-1]; cm != m || cn != n {
		return la, lb, lc, 0, errors.Errorf(
			"MatMulBatched: result matrix shape is (%d, %d) but (%d, %d) is required", cm, cn, m, n)
	}

	// BLAS can only address one storage order per call; mixing row- and
	// col-major operands has no stride representation for a single GEMM. The
	// 2D path has the same limitation.
	order := a.DataOrder()
	if !b.DataOrder().HasSameOrder(order) || !c.DataOrder().HasSameOrder(order) {
		return la, lb, lc, 0, errors.Errorf(
			"MatMulBatched: all operands must share the same data order. "+
				"Got a: %v, b: %v, ret: %v", a.DataOrder(), b.DataOrder(), c.DataOrder())
	}

	// In col-major storage the leading (outer, shape[0]) batch axis is the
	// fastest-varying dimension: successive batch matrices are interleaved
	// element-by-element, so no per-batch window has the fixed leading
	// dimension BLAS requires. Addressing them in place would demand a
	// strided-batch GEMM that the BLAS interface here does not provide. Rather
	// than silently reordering (copying) the input, the layout is rejected.
	// Row-major storage keeps each batch matrix in a contiguous outer window,
	// which is the layout the batched path addresses directly.
	if order.IsColMajor() {
		return la, lb, lc, 0, errors.Errorf(
			"MatMulBatched: col-major tensors with a batch dimension (shape %v) interleave their batches and cannot be addressed in place by BLAS; materialize or use row-major storage",
			a.Shape())
	}

	if la, err = matrixLayout(a, batchA); err != nil {
		return la, lb, lc, 0, errors.Wrap(err, "MatMulBatched: operand a is not usable")
	}
	if lb, err = matrixLayout(b, batchB); err != nil {
		return la, lb, lc, 0, errors.Wrap(err, "MatMulBatched: operand b is not usable")
	}
	if lc, err = matrixLayout(c, batchC); err != nil {
		return la, lb, lc, 0, errors.Wrap(err, "MatMulBatched: result buffer is not usable")
	}

	// Bounds-check every window BLAS will touch. A broadcast operand repeats
	// its single matrix and never walks past it.
	if err = checkLayoutBounds("a", a, la, batches, batchA); err != nil {
		return la, lb, lc, 0, err
	}
	if err = checkLayoutBounds("b", b, lb, batches, batchB); err != nil {
		return la, lb, lc, 0, err
	}
	if err = checkLayoutBounds("c", c, lc, batches, batchC); err != nil {
		return la, lb, lc, 0, err
	}
	return la, lb, lc, batches, nil
}

// matrixLayout derives a matLayout from the two innermost dimensions of t.
//
// The batched path operates on row-major storage, where every batch matrix
// occupies a contiguous outer window. A per-batch matrix can only be handed to
// BLAS in one of two physical layouts:
//
//   - a standard row-major (rows, cols) matrix with strides (cols, 1);
//   - the same memory read as its transpose, with strides (1, rows).
//
// Any other inner stride pattern (for example a view produced by slicing an
// inner dimension with step > 1) cannot be expressed to BLAS; it is rejected
// rather than silently materialized. Negative strides are likewise rejected.
//
// Degenerate dimensions (size 0 or 1) make the two layouts indistinguishable;
// the no-transpose descriptor is then always correct.
func matrixLayout(t DenseTensor, batchSize int) (matLayout, error) {
	if t.Dims() < 2 {
		return matLayout{}, errors.Errorf("expected at least a matrix per batch, got shape %v", t.Shape())
	}
	strides := t.Strides()
	for _, s := range strides {
		if s < 0 {
			return matLayout{}, errors.Errorf(
				"strided view on shape %v has a negative stride %v; such a view cannot be addressed in place",
				t.Shape(), strides)
		}
	}

	d := t.Dims()
	rows := t.Shape()[d-2]
	cols := t.Shape()[d-1]
	// The two innermost strides describe the per-batch matrix; when a batch
	// axis exists it sits at d-3 and its stride is read below.
	rowStride := strides[d-2]
	colStride := strides[d-1]

	normal := cols <= 1 || (rowStride == cols && colStride == 1)
	transposed := rows <= 1 || (rowStride == 1 && colStride == rows)
	if !normal && !transposed {
		return matLayout{}, errors.Errorf(
			"strided view on shape %v with strides %v is not a standard or transposed dense matrix; such a view cannot be addressed in place",
			t.Shape(), strides)
	}
	trans := transposed && !normal

	var ld int
	if trans {
		ld = colStride
	} else {
		ld = rowStride
	}

	batchStride := 0
	if d >= 3 && batchSize > 1 {
		batchStride = strides[d-3]
	}
	return matLayout{
		rows:        rows,
		cols:        cols,
		ld:          ld,
		rowStride:   rowStride,
		colStride:   colStride,
		batchStride: batchStride,
		trans:       trans,
	}, nil
}

// checkLayoutBounds verifies that addressing every batch through the layout
// stays inside the tensor's own backing array. The offset of a sliced view is
// already baked into its backing slice, so element 0 is always the view's own
// first element.
//
// ownBatches is the operand's own batch dimension. When it is 1 the operand is
// broadcast: its single matrix is reused for every batch (batchStride 0) and
// the bounds check must only cover that one matrix.
func checkLayoutBounds(name string, t DenseTensor, l matLayout, batches, ownBatches int) error {
	// Empty matrices and empty batches address no elements.
	if l.rows == 0 || l.cols == 0 || batches == 0 {
		return nil
	}
	lastBatch := ownBatches - 1
	if lastBatch < 0 {
		lastBatch = 0
	}
	lastRow := l.rows - 1
	lastCol := l.cols - 1
	maxIndex := lastBatch*l.batchStride + lastRow*l.rowStride + lastCol*l.colStride
	avail := t.len()
	if maxIndex < 0 || maxIndex >= avail {
		return errors.Errorf(
			"MatMulBatched: view %s on shape %v with strides %v addresses element %d but the backing memory only holds %d elements",
			name, t.Shape(), t.Strides(), maxIndex, avail)
	}

	// Mirror BLAS's own leading-dimension requirements so the error message
	// describes the offending view instead of surfacing as a panic.
	if l.ld < 1 {
		return errors.Errorf(
			"MatMulBatched: view %s on shape %v with strides %v has an unusable leading dimension %d",
			name, t.Shape(), t.Strides(), l.ld)
	}
	return nil
}

// blasTranspose maps the layout's transpose marker onto a BLAS flag.
func (l matLayout) blasTranspose() blas.Transpose {
	if l.trans {
		return blas.Trans
	}
	return blas.NoTrans
}

// batchedDgemm runs one Dgemm per batch, addressing the caller's arrays
// directly. Re-slicing a slice changes only the slice header on the stack and
// allocates no heap memory.
func batchedDgemm(la, lb, lc matLayout, batches int, a, b, c []float64) {
	tA, tB := la.blasTranspose(), lb.blasTranspose()
	alpha, beta := float64(1), float64(0)
	m, n, k := la.rows, lb.cols, la.cols

	for i := 0; i < batches; i++ {
		aw := a[i*la.batchStride:]
		bw := b[i*lb.batchStride:]
		cw := c[i*lc.batchStride:]
		whichblas.Dgemm(tA, tB, m, n, k, alpha, aw, la.ld, bw, lb.ld, beta, cw, lc.ld)
	}
}

// batchedSgemm is the float32 counterpart of batchedDgemm.
func batchedSgemm(la, lb, lc matLayout, batches int, a, b, c []float32) {
	tA, tB := la.blasTranspose(), lb.blasTranspose()
	alpha, beta := float32(1), float32(0)
	m, n, k := la.rows, lb.cols, la.cols

	for i := 0; i < batches; i++ {
		aw := a[i*la.batchStride:]
		bw := b[i*lb.batchStride:]
		cw := c[i*lc.batchStride:]
		whichblas.Sgemm(tA, tB, m, n, k, alpha, aw, la.ld, bw, lb.ld, beta, cw, lc.ld)
	}
}

// memoryOverlap reports whether x and y share any backing memory. View offsets
// are baked into the byte slices, so comparing the [Ptr,Ptr+len) intervals
// covers offset views as well.
func memoryOverlap(x, y DenseTensor) bool {
	xh, yh := x.hdr(), y.hdr()
	if len(xh.Raw) == 0 || len(yh.Raw) == 0 {
		return false
	}
	xStart := uintptr(unsafe.Pointer(&xh.Raw[0]))
	yStart := uintptr(unsafe.Pointer(&yh.Raw[0]))
	xEnd := xStart + uintptr(len(xh.Raw))
	yEnd := yStart + uintptr(len(yh.Raw))
	return xStart < yEnd && yStart < xEnd
}

// ensureNonOverlappingOutput rejects configurations where the output shares
// backing memory with an input: GEMM semantics require C to be distinct from
// A and B, and overlapping windows would make the result order-dependent.
func ensureNonOverlappingOutput(a, b, out DenseTensor) error {
	if memoryOverlap(out, a) {
		return errors.Errorf("MatMulBatched: the result buffer overlaps the memory of operand a; in-place batched multiplication is not supported")
	}
	if memoryOverlap(out, b) {
		return errors.Errorf("MatMulBatched: the result buffer overlaps the memory of operand b; in-place batched multiplication is not supported")
	}
	return nil
}
