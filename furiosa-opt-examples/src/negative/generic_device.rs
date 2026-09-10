//! Generic device functions used by compile diagnostics and macro-expansion tests.

use furiosa_opt_std::prelude::*;

use crate::generic::B;

type Matrix<A> = HbmTensor<i32, m![1], m![A, B]>;

#[device(chip = 1)]
pub fn where_bound_copy<A>(device: &mut Device, hbm: &Matrix<A>) -> Matrix<A>
where
    A: AxisName,
{
    let dm: DmTensor<i32, m![1], m![1], m![A], m![B]> = hbm.to_dm(&mut device.tdma);
    dm.to_hbm(&mut device.tdma)
}

pub mod a {
    use super::*;

    #[device(chip = 1)]
    pub fn gen_copy<A: AxisName>(
        device: &mut Device,
        hbm: &HbmTensor<i32, m![1], m![A, B]>,
    ) -> HbmTensor<i32, m![1], m![A, B]> {
        let dm: DmTensor<i32, m![1], m![1], m![A], m![B]> = hbm.to_dm(&mut device.tdma);
        dm.to_hbm(&mut device.tdma)
    }
}

pub mod b {
    use super::*;

    #[device(chip = 1)]
    pub fn gen_copy<A: AxisName>(
        device: &mut Device,
        hbm: &HbmTensor<i32, m![1], m![A, B]>,
    ) -> HbmTensor<i32, m![1], m![A, B]> {
        let dm: DmTensor<i32, m![1], m![1], m![A], m![B]> = hbm.to_dm(&mut device.tdma);
        dm.to_hbm(&mut device.tdma)
    }
}
