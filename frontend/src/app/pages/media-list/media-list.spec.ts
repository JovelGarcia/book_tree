import { ComponentFixture, TestBed } from '@angular/core/testing';

import { MediaList } from './media-list';

describe('MediaList', () => {
  let component: MediaList;
  let fixture: ComponentFixture<MediaList>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [MediaList]
    })
    .compileComponents();

    fixture = TestBed.createComponent(MediaList);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
