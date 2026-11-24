package tensor

import "github.com/pkg/errors"

// This file adds batched matrix multiplication to *Dense. It sits above the
// engine exactly like MatMul: shape validation, reuse/incr handling and output
// allocation live here, while the engine only performs strided BLAS calls on
// already-validated tensors.
//
// The 2D MatMul path is deliberately untouched: its dispatch, transpose
// handling and buffer-reuse semantics remain exactly as they were.

// MatMulBatched multiplies two tensors that each carry a leading batch
// dimension.
//
// a has shape (ba, m, k) and b has shape (bb, k, n). The batch dimensions ba
// and bb must either be equal or one of them must be 1 (broadcast); the result
// has shape (max(ba, bb), m, n). Every batch is computed as an independent 2D
// matrix multiplication, so the result is element-wise equal to looping over
// the batches and calling MatMul on each pair of 2D matrices, including the
// propagation of NaN/Inf values.
//
// The operands are addressed in their own memory through their strides.
// Transposed views, offset views produced by slicing and other non-contiguous
// layouts are supported without copying; views that cannot be expressed to
// BLAS (negative strides or out-of-bounds addressing) are rejected with an
// error rather than silently materialized.
//
// The usual FuncOpt options apply: WithReuse writes into an existing result
// tensor and WithIncr accumulates into an existing tensor. When WithIncr is
// used, the destination's current values genuinely participate in the
// computation (the batched product is added to them); they are never silently
// zeroed.
func (t *Dense) MatMulBatched(other Tensor, opts ...FuncOpt) (retVal *Dense, err error) {
	// Exactly one leading batch axis plus a 2D matrix is in scope.
	if t.Dims() != 3 || other.Dims() != 3 {
		return nil, errors.Errorf(
			"MatMulBatched supports exactly one batch dimension plus a matrix (3D operands). "+
				"Got t's shape: %v, other's shape: %v", t.Shape(), other.Shape())
	}

	od, ok := other.(*Dense)
	if !ok {
		return nil, errors.Errorf("MatMulBatched requires both operands to be *Dense. Got %T instead", other)
	}

	// dtype check, restricted to the two real floating point kinds the
	// batched BLAS path supports.
	if t.Dtype() != od.Dtype() {
		return nil, errors.Errorf(typeMismatch, t.Dtype(), od.Dtype())
	}
	switch t.Dtype() {
	case Float32, Float64:
	default:
		return nil, errors.Errorf("MatMulBatched only supports Float32 and Float64. Got %v instead", t.Dtype())
	}

	batchA := t.Shape()[0]
	batchB := od.Shape()[0]
	batches, err := broadcastBatches(batchA, batchB, t.Shape(), od.Shape())
	if err != nil {
		return nil, err
	}

	m, k1 := t.Shape()[1], t.Shape()[2]
	k2, n := other.Shape()[1], other.Shape()[2]
	if k1 != k2 {
		return nil, errors.Errorf(shapeMismatch, t.Shape(), od.Shape())
	}
	expectedShape := Shape{batches, m, n}

	fo := ParseFuncOpts(opts...)
	defer returnOpOpt(fo)
	if retVal, err = handleReuse(fo.Reuse(), expectedShape, fo.Safe()); err != nil {
		return nil, errors.Wrapf(err, opFail, "MatMulBatched")
	}

	if retVal == nil {
		// The batched path addresses row-major batch windows, so the result is
		// always row-major; col-major inputs are rejected by the engine below.
		retVal = recycledDense(t.t, expectedShape, WithEngine(t.e))
	}

	// GEMM requires the output not to alias its inputs. This is checked here
	// (before any work happens) and reported as an error instead of being
	// left to BLAS, whose behavior on overlapping operands is undefined.
	if err = ensureNonOverlappingOutput(t, od, retVal); err != nil {
		return nil, err
	}
	if incr := fo.Incr(); incr != nil {
		incrD, ierr := assertDense(incr)
		if ierr != nil {
			return nil, errors.Wrapf(ierr, opFail, "MatMulBatched")
		}
		if err = ensureNonOverlappingOutput(t, od, incrD); err != nil {
			return nil, err
		}
	}

	e := t.e
	mm, ok := e.(BatchedMatMuler)
	if !ok {
		return nil, errors.New("engine does not support MatMulBatched")
	}
	if err = mm.MatMulBatched(t, od, retVal); err != nil {
		return nil, err
	}
	return handleIncr(retVal, fo.Reuse(), fo.Incr(), expectedShape)
}

// broadcastBatches applies the batch-dimension broadcast rules and reports the
// two batch shapes in the error message for the non-broadcastable case.
func broadcastBatches(batchA, batchB int, shapeA, shapeB Shape) (int, error) {
	switch {
	case batchA == batchB:
		return batchA, nil
	case batchA == 1:
		return batchB, nil
	case batchB == 1:
		return batchA, nil
	default:
		return 0, errors.Errorf(
			"MatMulBatched: batch dimensions cannot be broadcast: a's batch shape is %v, b's batch shape is %v",
			shapeA[:1], shapeB[:1])
	}
}
