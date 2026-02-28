import { ComponentFixture, TestBed } from '@angular/core/testing';

import { EpubRelationships } from './epub-relationships';

describe('EpubRelationships', () => {
  let component: EpubRelationships;
  let fixture: ComponentFixture<EpubRelationships>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [EpubRelationships]
    })
    .compileComponents();

    fixture = TestBed.createComponent(EpubRelationships);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
