import { ComponentFixture, TestBed } from '@angular/core/testing';

import { MediaSearch } from './media-search';

describe('MediaSearch', () => {
  let component: MediaSearch;
  let fixture: ComponentFixture<MediaSearch>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [MediaSearch]
    })
    .compileComponents();

    fixture = TestBed.createComponent(MediaSearch);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
