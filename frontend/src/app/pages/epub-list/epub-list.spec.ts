import { ComponentFixture, TestBed } from '@angular/core/testing';

import { EpubList } from './epub-list';

describe('EpubList', () => {
  let component: EpubList;
  let fixture: ComponentFixture<EpubList>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [EpubList]
    })
    .compileComponents();

    fixture = TestBed.createComponent(EpubList);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
