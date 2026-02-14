import { ComponentFixture, TestBed } from '@angular/core/testing';

import { EpubDetail } from './epub-detail';

describe('EpubDetail', () => {
  let component: EpubDetail;
  let fixture: ComponentFixture<EpubDetail>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [EpubDetail]
    })
    .compileComponents();

    fixture = TestBed.createComponent(EpubDetail);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
