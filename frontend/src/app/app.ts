import { Component } from '@angular/core';
import { MediaSearch } from './pages/media-search/media-search';

@Component({
  selector:   'app-root',
  standalone: true,
  imports:    [MediaSearch],
  template:   `<app-media-search />`,
})
export class App {}
